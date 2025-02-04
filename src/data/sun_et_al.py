import random

import numpy as np
import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset, Sampler

from src.helper_function import circular_slice
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
    ):
        """Constructor for the MicrobiomeDataset class.

        Args:
            samples (pd.DataFrame): A dataframe containing the samples as rows and the features as columns. The dataframe
                must have the same index as the label_and_project dataframe, representing the samples.
            label_and_project (pd.DataFrame): A dataframe containing the labels and the project of each sample.
                The dataframe must have the same index as the samples dataframe. The dataframe must have two columns:
                "label" and "project". The "label" column contains the labels of the samples and the "project" column
                contains the project each sample belongs to. The dataframe must have the same index as the samples dataframe
                representing the samples.
            preprocessor (callable, optional): A callable that preprocesses the samples dataframe. Defaults to None.
            target_preprocessor (callable, optional): A callable that preprocesses the label_and_project dataframe. Defaults
                to None. # TODO default should at least encode the labels to integers
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
        # sort the samples dataframe by index and reset the index
        self.samples = samples.sort_index().reset_index(drop=True)
        if preprocessor is not None:
            self.samples = preprocessor(self.samples)
        self.samples = Tensor(self.samples.to_numpy())

        label_and_project = label_and_project.sort_index().reset_index(drop=True)
        if target_preprocessor is not None:
            label_and_project = target_preprocessor(label_and_project)

        by_project_grouped_labels = label_and_project.groupby("project")[["label"]]
        # Create a dictionary that maps the project to the indices per class for that project
        self.group_to_label_idx_per_class = {
            group: {
                label: np.array(indices)    # np arrays used to make circular slicing for efficient
                for label, indices in group_df.groupby("label").groups.items()
            }
            for group, group_df in by_project_grouped_labels
        }
        self.labels = Tensor(label_and_project["label"].to_numpy())

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
        if self.transform:
            sample = self.transform(sample)

        if self.target_transform:
            label = self.target_transform(label)

        return sample, label

    def get_group_indices(self, group: object) -> list[int]:
        """Returns the indices of the samples that belong to the given group.

        Args:
            group (str): The name of the group.

        Returns:
            list: The indices of the samples that belong to the given group.
        """
        return self.group_to_label_idx_per_class[group]

    def get_groups(self) -> list[object]:
        """Returns the names of the groups in the dataset.

        Returns:
            list: The names of the groups in the dataset.
        """
        return list(self.groups.keys())


class BinaryFewShotBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset: MicrobiomeDataset,
        K_shot: int,
        include_query: bool = False,
        shuffle: bool = True,
        shuffle_once: bool = True,
    ):
        self.K_shot = K_shot
        self.include_query = include_query
        self.shuffle = shuffle
        self.shuffle_once = shuffle_once
        self.batch_size = self.K_shot * 2  # 2 classes
        if self.include_query:
            self.batch_size *= 2

        self.dataset = dataset
        self.groups = list(dataset.group_to_label_idx_per_class.keys())

        # number of batches each group can give is given by its largest class (so we oversample the smaller class)
        # +1 for each group to make sure all data is sampled in each epoch (mainly to prevent issues with large K_shot and small classes)
        self.batches_per_group = {
            group: max(len(ids) for ids in label_dict.values()) // self.K_shot + 1
            for group, label_dict in dataset.group_to_label_idx_per_class.items()
        }

        self.groups_to_sample = [
            group
            for group, n_batches in self.batches_per_group.items()
            for _ in range(n_batches)
        ]

        if self.shuffle_once or self.shuffle:
            self.shuffle_data()
        else:
            # for testing
            pass

    def shuffle_data(self):
        # TODO think about this: because for validation shuffle is only done once, a some data will just not be selected for the smaller class
        random.shuffle(self.groups_to_sample)
        for group in self.groups:
            for label, label_ids in self.dataset.group_to_label_idx_per_class[
                group
            ].items():
                np.random.shuffle(label_ids)

    def __iter__(self):
        if self.shuffle:
            self.shuffle_data()

        start_indices_per_group = {group: 0 for group in self.groups}
        for group in self.groups_to_sample:
            batch = []
            label_ids_per_class = self.dataset.group_to_label_idx_per_class[group]
            start_idx = start_indices_per_group[group]
            for label_ids in label_ids_per_class.values():
                # We sample the label_ids of the class circularly as we have number of batches based on the
                # largest class for each group (so the smaller class is oversampled)
                batch.extend(circular_slice(label_ids, start_idx, start_idx + self.K_shot))
            start_indices_per_group[group] += self.K_shot
            start_idx = start_indices_per_group[group]
            if self.include_query:
                for label_ids in label_ids_per_class.values():
                    # We sample the label_ids of the class circularly as we have number of batches based on the
                    # largest class for each group (so the smaller class is oversampled)
                    batch.extend(circular_slice(label_ids, start_idx, start_idx + self.K_shot))
                start_indices_per_group[group] += self.K_shot
            yield batch

    def __len__(self):
        return len(self.groups_to_sample)
