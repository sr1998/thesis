import random
from time import sleep

from loguru import logger
import numpy as np
import pandas as pd
from torch import float32, tensor, randn_like
from torch.utils.data import Dataset, Sampler
import torch

from src.helper_function import circular_slice, df_str_for_loguru
from src.preprocessing.functions import pandas_label_encoder
from src.global_vars import RANDOM_SEED


class MicrobiomeDataset(Dataset):
    """..."""

    def __init__(
        self,
        samples: pd.DataFrame,
        label_and_project: pd.DataFrame,
        preprocessor=None,
        target_preprocessor=pandas_label_encoder,
        transform=None,
        target_transform=None,
        preselected_support_set: list[str] = None,  # only for val and test
        jitter_fraction: float = 0.0,       # 10 % of within‑study σ  (tune as you like)
        device: str = "cpu",
    ):
        """Constructor for the MicrobiomeDataset class.

        Args:
            samples (pd.DataFrame): A dataframe containing the samples as rows and the features as columns. The dataframe
                must have the same index as the label_and_project dataframe, representing the samples.
            label_and_project (pd.DataFrame): A dataframe containing the label and the project of each sample.
                The dataframe must have the same index as the samples dataframe. The dataframe must have two columns:
                "label" and "project". The "label" column contains the label for each samples and the "project" column
                contains the project each sample belongs to. The dataframe must have the same index as the samples dataframe
                representing the samples.
            preprocessor (callable, optional): A callable that preprocesses the samples dataframe. Defaults to None.
            target_preprocessor (callable, optional): A callable that preprocesses the label_and_project dataframe. Defaults
                to None.
            transform (callable, optional): A callable that transforms the samples. Defaults to None.
            target_transform (callable, optional): A callable that transforms the targets. Defaults to None. Due to meta-
                learning, the labels are not transformed.
        """
        super().__init__()
        # assert indices being the same for both dataframes
        assert set(samples.index) == set(
            label_and_project.index
        ), "Indices of both dataframes must be the same"

        self.samples = samples
        self.device = device

        self.preselected_support_set_used = bool(preselected_support_set)
        if self.preselected_support_set_used:
            # Order samples in such a way that the support set is always on top and selected with the corresponding k_shot selection.
            # So first n_classes*k_shot samples should be the support set for each group when grouped by class, as there are this many support set indices.
            self.samples = self.samples.loc[preselected_support_set + [idx for idx in self.samples.index if idx not in preselected_support_set]]

        # Make samples and labels have the same order
        label_and_project = label_and_project.loc[self.samples.index] # TODO check for full dataset that is already sorted

        # Reset the index
        self.samples = self.samples.reset_index(drop=True)
        raw_samples_df = self.samples.copy()

        if preprocessor is not None:
            self.samples = preprocessor(self.samples)
        self.samples = tensor(self.samples.to_numpy(), dtype=float32)

        # we can drop here, as we already selected the indices desired
        label_and_project = label_and_project.reset_index(drop=True)
        if target_preprocessor is not None:
            label_and_project = target_preprocessor(label_and_project)

        by_project_grouped_labels = label_and_project.groupby("project", sort=False)[["label"]]
        # Create a dictionary that maps the project to the indices per class for that project
        self.group_to_label_idx_per_class = {
            group: {
                label: np.array(indices)    # np arrays used to make circular slicing efficient
                for label, indices in group_df.groupby("label", sort=False).groups.items()
            }
            for group, group_df in by_project_grouped_labels
        }
        # labels, sorted as samples
        self.projects = label_and_project["project"].tolist()
        self.labels = tensor(label_and_project["label"].to_numpy(), dtype=float32)

        self.jitter_fraction = jitter_fraction
        if self.jitter_fraction:
            self.project_sigma = {}
            grouped = raw_samples_df.groupby(label_and_project["project"])
            for proj, df in grouped:
                sd = df.std(axis=0)                                 # pandas per‑feature s.d.
                # fall back to tiny ε so we never divide by zero
                sd_tensor = tensor(sd.fillna(0.0).to_numpy(), dtype=float32)
                self.project_sigma[proj] = sd_tensor * jitter_fraction

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        """Returns the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx: int):
        """Returns the sample and the label at the given index with the transformations applied.

        Args:
            idx (int): The index of the sample and label to return.

        """
        sample = self.samples[idx]
        label = self.labels[idx]
        proj = self.projects[idx]

        if self.jitter_fraction:
            sigma_vec = self.project_sigma[proj]
            noise = randn_like(sample) * sigma_vec
            sample = sample + noise

        if self.transform:
            sample = self.transform(sample)

        if self.target_transform:
            label = self.target_transform(label)

        # sample = sample.to(self.device)
        # label = label.to(self.device)

        return sample, label

    # def __getitems__(self, indices: list[int]) -> list[tensor, tensor]:
    #     """Returns the samples and labels at the given indices with the transformations applied.
        
    #      To be used with collate_fn lambda x: x

    #     Args:
    #         indices (list[int]): The indices of the samples and labels to return.

    #     """
    #     samples = self.samples[indices]
    #     labels = self.labels[indices]
    #     if self.transform:
    #         samples = self.transform(samples)

    #     if self.target_transform:
    #         labels = self.target_transform(labels)

    #     return [samples, labels]


class BinaryFewShotBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset: MicrobiomeDataset,
        k_shot: int,
        include_query: bool = True,
        shuffle: bool = True,
        shuffle_once: bool = True,
        training: bool = True,
    ):
        self.k_shot = k_shot
        self.include_query = include_query or not training  # Always include query set for validation/testing
        self.shuffle = shuffle
        self.shuffle_once = shuffle_once
        # self.batch_size = self.k_shot * 2  # 2 classes
        # if self.include_query:
        #     self.batch_size *= 2

        self.dataset = dataset
        self.groups = list(dataset.group_to_label_idx_per_class.keys())
        self.training = training

        self.cached_eval_batches = None
        self.cache_eval_batches = True

        if self.training:
            # number of batches each group can give is given by its largest class (so we oversample the smaller class)
            # +1 for each group to make sure all data is sampled in each epoch (mainly to prevent issues with large K_shot and small classes)
            n_batches_per_group_training = {
                group: max(len(ids) for ids in label_dict.values()) // self.k_shot + 1
                for group, label_dict in dataset.group_to_label_idx_per_class.items()
            }

            self.groups_to_sample_training = [
                group
                for group, n_batches in n_batches_per_group_training.items()
                for _ in range(n_batches)
            ]

            if self.shuffle_once or self.shuffle:
                random.shuffle(self.groups_to_sample_training)
                # for group in self.groups:
                #     for label, label_ids in self.dataset.group_to_label_idx_per_class[
                #         group
                #     ].items():
                #         np.random.shuffle(label_ids)

        # validation/testing:
        # Each group is sampled once and the query set is all the data except the support set.
        # n_batches_per_group_eval = {group: 1 for group in self.groups}
        self.groups_to_sample_eval = self.groups.copy()


        if (self.shuffle_once or self.shuffle_once):
            for group in self.groups:
                    for label, label_ids in self.dataset.group_to_label_idx_per_class[
                        group
                    ].items():
                        np.random.shuffle(label_ids)
            if not self.dataset.preselected_support_set_used:
                random.shuffle(self.groups_to_sample_eval) 

    def __iter__(self):
        # For eval data with caching
        if not self.training and self.cached_eval_batches is not None and self.cache_eval_batches:
            for batch in self.cached_eval_batches:
                yield batch
            return

        if self.training:    # No shuffle if preselected support set is used
            random.shuffle(self.groups_to_sample_training)
            for group in self.groups:
                    for label, label_ids in self.dataset.group_to_label_idx_per_class[
                        group
                    ].items():
                        np.random.shuffle(label_ids)

        start_indices_per_group = {group: 0 for group in self.groups}
        if self.training:
            for group in self.groups_to_sample_training: # group gives a task
                batch = []
                label_ids_per_class = self.dataset.group_to_label_idx_per_class[group]
                start_idx = start_indices_per_group[group]
                for label_ids in label_ids_per_class.values():
                    # We sample the label_ids of the class circularly as we have number of batches based on the
                    # largest class for each group (so the smaller class is oversampled)
                    # Circular slicing like this can be a problem if the number of samples is less than 2*k_shot,
                    # as query and support set will overlap. So, be mindful in case you need both.
                    batch.extend(circular_slice(label_ids, start_idx, start_idx + self.k_shot))
                start_indices_per_group[group] += self.k_shot
                start_idx = start_indices_per_group[group]
                if self.include_query:
                    for label_ids in label_ids_per_class.values():
                        # We sample the label_ids of the class circularly as we have number of batches based on the
                        # largest class for each group (so the smaller class is oversampled)
                        # Circular slicing like this can be a problem if the number of samples is less than 2*k_shot,
                        # as query and support set will overlap. So, be mindful in case you need both.
                        batch.extend(circular_slice(label_ids, start_idx, start_idx + self.k_shot))
                        start_indices_per_group[group] += self.k_shot

                yield batch
        else:
            all_batches = []
            for group in self.groups_to_sample_eval: # group gives a task
                batch = []
                label_ids_per_class = self.dataset.group_to_label_idx_per_class[group]
                start_idx = start_indices_per_group[group]
                for label_ids in label_ids_per_class.values():
                    # We sample the label_ids of the class circularly as we have number of batches based on the
                    # largest class for each group (so the smaller class is oversampled)
                    # Circular slicing like this can be a problem if the number of samples is less than 2*k_shot,
                    # as query and support set will overlap. So, be mindful in case you need both.
                    batch.extend(circular_slice(label_ids, start_idx, start_idx + self.k_shot))
                start_indices_per_group[group] += self.k_shot
                start_idx = start_indices_per_group[group]
                if self.include_query:
                    for label_ids in label_ids_per_class.values():
                        new_idx = label_ids[start_idx:]
                        if new_idx is None or len(new_idx) == 0:
                            new_idx = circular_slice(label_ids, start_idx, start_idx + self.k_shot)
                        batch.extend(new_idx)
                        # start_indices_per_group[group] += len(label_ids[start_idx:])    # not necessary as each group is sampled once
                    
                all_batches.append(batch)

            if self.cache_eval_batches:
                self.cached_eval_batches = all_batches

            for batch in all_batches:
                yield batch

    def __len__(self):
        if self.training:
            return len(self.groups_to_sample_training)
        else:
            return len(self.groups_to_sample_eval)
    

class LabelOnlyDataset(Dataset):
    """Dataset class that only distinguishes between labels without separating studies."""

    def __init__(
        self,
        samples,
        labels,
        preprocessor=None,
        transform=None,
        target_transform=None,
        device='cpu',
    ):
        """Constructor for the LabelOnlyDataset class.

        Args:
            samples: A numpy array containing the samples as rows and features as columns.
            labels: A numpy array containing the labels for each sample.
            preprocessor (callable, optional): Preprocesses the samples. Defaults to None.
            transform (callable, optional): Transforms the samples. Defaults to None.
            target_transform (callable, optional): Transforms the labels. Defaults to None.
        """
        super().__init__()
        
        # Apply preprocessor if provided
        if preprocessor is not None:
            samples = preprocessor(self.samples)

        # Convert to tensors and move to device in constructor
        if isinstance(samples, torch.Tensor):
            self.samples = samples.to(device)
        else:
            self.samples = torch.tensor(samples, dtype=torch.float32).to(device)
            
        if isinstance(labels, torch.Tensor):
            self.labels = labels.to(device)
        else:
            self.labels = torch.tensor(labels, dtype=torch.float32).to(device)
        
        # Store indices by label for efficient sampling
        self.indices_by_label = {}
        for idx, label in enumerate(self.labels):
            label_val = label.item()
            if label_val not in self.indices_by_label:
                self.indices_by_label[label_val] = []
            self.indices_by_label[label_val].append(idx)
            
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Returns the sample and label at the given index with transformations applied."""
        sample = self.samples[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
            
        if self.target_transform:
            label = self.target_transform(label)
            
        return sample, label

class KShotBatchSampler(Sampler):
    """K-shot batch sampler that creates batches with k samples from each class,
    undersampling the larger class during training."""

    def __init__(
        self,
        dataset: LabelOnlyDataset,
        k_shot: int,
        include_query: bool = True,
        query_size: int = 0,
        shuffle: bool = True,
    ):
        """Initialize the KShotBatchSampler.
        
        Args:
            dataset: Dataset containing samples and indices by label
            k_shot: Number of samples to include per class in support set
            include_query: Whether to include a query set in batches
            query_size: Number of samples per class in query set (defaults to k_shot if None)
                        Special value "rest" will use all remaining samples
            shuffle: Whether to shuffle the samples
        """
        self.dataset = dataset
        self.k_shot = k_shot
        self.include_query = include_query
        self.use_all_remaining = query_size == "rest"
        self.query_size = query_size if not self.use_all_remaining else 0
        if self.query_size == 0 and not self.use_all_remaining:
            self.query_size = k_shot
        self.shuffle = shuffle
        
        # Check if we have enough samples per class for k_shot
        for label, indices in self.dataset.indices_by_label.items():
            if len(indices) < k_shot:
                raise ValueError(f"Label {label} has fewer samples ({len(indices)}) than required for k_shot ({k_shot})")
        
        # Check if we have enough samples for both support and query sets during training
        min_required = k_shot
        min_required += self.query_size
        if include_query and not self.use_all_remaining:
            for label, indices in self.dataset.indices_by_label.items():
                if len(indices) < min_required:
                    raise ValueError(f"Label {label} has fewer samples ({len(indices)}) than required for support+query ({min_required})")

    def __iter__(self):
        """Yields batches of indices for k-shot sampling with support and query sets."""
        # Copy the indices to avoid modifying the original
        indices_by_label = {
            label: indices.copy() for label, indices in self.dataset.indices_by_label.items()
        }
        
        # Shuffle if needed
        if self.shuffle:
            for indices in indices_by_label.values():
                random.shuffle(indices)
        
        # Determine number of batches based on smallest class size
        min_class_size = min(len(indices) for indices in indices_by_label.values())
        
        if self.use_all_remaining:
            # When using "rest", we can only have 1 batch per full dataset
            n_batches = 1
        else:
            # Otherwise calculate based on smallest class size to ensure undersampling
            samples_per_batch_per_class = self.k_shot
            if self.include_query:
                samples_per_batch_per_class += self.query_size
            n_batches = min_class_size // samples_per_batch_per_class
        
        # Generate batches
        for batch_idx in range(n_batches):
            batch = []
            
            # First add support set (k samples per class)
            for label, indices in indices_by_label.items():
                if self.use_all_remaining:
                    support_indices = indices[:self.k_shot]
                else:
                    start_idx = batch_idx * (self.k_shot + (self.query_size if self.include_query else 0))
                    support_indices = indices[start_idx:start_idx + self.k_shot]
                batch.extend(support_indices)
            
            # Then add query set if required
            if self.include_query:
                for label, indices in indices_by_label.items():
                    if self.use_all_remaining:
                        query_indices = indices[self.k_shot:]
                    else:
                        start_idx = batch_idx * (self.k_shot + self.query_size) + self.k_shot
                        query_indices = indices[start_idx:start_idx + self.query_size]
                    batch.extend(query_indices)
            
            yield batch

    def __len__(self):
        """Returns the number of batches."""
        if self.use_all_remaining:
            return 1  # Only one batch when using "rest"
        else:
            min_class_size = min(len(indices) for indices in self.dataset.indices_by_label.values())
            samples_per_batch_per_class = self.k_shot
            if self.include_query:
                samples_per_batch_per_class += self.query_size
            return min_class_size // samples_per_batch_per_class