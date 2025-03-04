import random
from time import sleep

from loguru import logger
import numpy as np
import pandas as pd
from torch import float32, tensor
from torch.utils.data import Dataset, Sampler

from src.helper_function import circular_slice, df_str_for_loguru
from src.preprocessing.functions import pandas_label_encoder


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
        preselected_support_set: list[str] = None,  # only for val and test#TODO test thoroughly
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

        self.preselected_support_set_used = bool(preselected_support_set)
        if self.preselected_support_set_used:
            # Order samples in such a way that the support set is always on top and selected with the corresponding k_shot selection.
            # So first n_classes*k_shot samples should be the support set for each group when grouped by class, as there are this many support set indices.
            self.samples = self.samples.loc[preselected_support_set + [idx for idx in self.samples.index if idx not in preselected_support_set]]

        # Make samples and labels have the same order
        label_and_project = label_and_project.loc[self.samples.index] # TODO check for full dataset that is already sorted

        # Reset the index
        self.samples = self.samples.reset_index(drop=True)

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
        self.labels = tensor(label_and_project["label"].to_numpy(), dtype=float32)

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
        samples = self.samples[idx]
        labels = self.labels[idx]
        if self.transform:
            samples = self.transform(samples)

        if self.target_transform:
            labels = self.target_transform(labels)

        return samples, labels


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

        if self.training:
            # number of batches each group can give is given by its largest class (so we oversample the smaller class)
            # +1 for each group to make sure all data is sampled in each epoch (mainly to prevent issues with large K_shot and small classes)
            self.n_batches_per_group = {
                group: max(len(ids) for ids in label_dict.values()) // self.k_shot + 1
                for group, label_dict in dataset.group_to_label_idx_per_class.items()
            }

            self.groups_to_sample = [
                group
                for group, n_batches in self.n_batches_per_group.items()
                for _ in range(n_batches)
            ]

            if self.shuffle_once or self.shuffle:
                self.shuffle_data()
        else:
            # validation/testing:
            # Each group is sampled once and the query set is all the data except the support set.
            self.n_batches_per_group = {group: 1 for group in self.groups}
            self.groups_to_sample = self.groups.copy()  # compatibility with training, but no effect

            if (self.shuffle_once or self.shuffle_once) and not self.dataset.preselected_support_set_used:
                self.shuffle_data()

    def shuffle_data(self):
        random.shuffle(self.groups_to_sample)
        for group in self.groups:
            for label, label_ids in self.dataset.group_to_label_idx_per_class[
                group
            ].items():
                np.random.shuffle(label_ids)

    def __iter__(self):
        if self.shuffle and not self.dataset.preselected_support_set_used:    # No shuffle if preselected support set is used
            self.shuffle_data()

        start_indices_per_group = {group: 0 for group in self.groups}
        for group in self.groups_to_sample: # group gives a task
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
                    if self.training:
                        # We sample the label_ids of the class circularly as we have number of batches based on the
                        # largest class for each group (so the smaller class is oversampled)
                        # Circular slicing like this can be a problem if the number of samples is less than 2*k_shot,
                        # as query and support set will overlap. So, be mindful in case you need both.
                        batch.extend(circular_slice(label_ids, start_idx, start_idx + self.k_shot))
                        start_indices_per_group[group] += self.k_shot
                    else:
                        batch.extend(label_ids[start_idx:])
                        # start_indices_per_group[group] += len(label_ids[start_idx:])    # not necessary as each group is sampled once
                
            yield batch

    def __len__(self):
        return len(self.groups_to_sample)
