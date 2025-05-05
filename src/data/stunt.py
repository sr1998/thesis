import os
import random
from re import I
import subprocess
import sys
import numpy as np
import pandas as pd
import copy
import psutil
from sklearn.cluster import KMeans
from src.global_vars import BASE_DATA_DIR, RANDOM_SEED
import multiprocessing as mp
from functools import partial

import copy

import numpy as np
from sklearn.cluster import KMeans

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    print("psutil not installed - attempting to install it...")

    subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
    import psutil
    PSUTIL_AVAILABLE = True
    print("psutil successfully installed and imported.")

def set_seed(seed=RANDOM_SEED):
    # Python's built-in random module
    random.seed(seed)
    
    # Numpy's random module
    np.random.seed(seed)


def microbiome_stunt_data_creator(unlabeled_x: pd.DataFrame, n_tasks, n_data_points, train_or_val, project_name, num_way=2, min_col_ratio=0.4, max_col_ratio=0.7, metadata=None):
    # Inspired by https://github.com/jaehyun513/STUNT/blob/main/data/income.py 
    n_data_points_per_task = int(n_data_points // n_tasks)
    # we want to keep the generated classes balanced for training; so we generate same number of points per class
    n_data_points_per_class = int(n_data_points_per_task // num_way)
    
    if train_or_val == "train":
        x = np.array(unlabeled_x)
        num_way = num_way
    elif train_or_val == "val":
        ...

    new_x = []
    new_y = []
    for t in range(n_tasks):

        if train_or_val == "val":
            ...
        elif train_or_val == "train":
            tmp_x = copy.deepcopy(x)
            min_count = 0
            while min_count < n_data_points_per_class:
                min_col = int(x.shape[1] * min_col_ratio)
                max_col = int(x.shape[1] * max_col_ratio)
                n_col = np.random.choice(range(min_col, max_col), 1, replace = False)[0]
                col_idx_to_select = np.random.choice([i for i in range(x.shape[1])], n_col, replace = False)
                selected_columns_to_get_labels = np.ascontiguousarray(x[:, col_idx_to_select], dtype = np.float32)

                kmeans = KMeans(num_way, max_iter=20, n_init=1)
                kmeans.fit(selected_columns_to_get_labels)
                y = kmeans.labels_

                class_list, counts = np.unique(y, return_counts = True)
                min_count = min(counts)
            
            num_to_permute = x.shape[0]
            for t_idx in col_idx_to_select:
                rand_perm = np.random.permutation(num_to_permute)
                tmp_x[:, t_idx] = tmp_x[:, t_idx][rand_perm]

            datapoint_idx = []
            for k in class_list:
                k_idx = np.where(y == k)[0]
                permutation = np.random.permutation(len(k_idx))
                k_idx = k_idx[permutation]
                datapoint_idx.append(k_idx[:n_data_points_per_class])
            datapoint_idx = np.concatenate(datapoint_idx)

            selected_x = tmp_x[datapoint_idx]
            selected_y = y[datapoint_idx]
            
            i = 0
            for k in class_list:
                selected_y[selected_y == k] = i
                i += 1


        new_x.append(pd.DataFrame(selected_x, columns=unlabeled_x.columns))

        selected_y = pd.DataFrame(selected_y, columns=["Group"]).map(lambda x: "Control" if x == 0 else "Disease")
        project_df = pd.DataFrame([f"{project_name}__{t}"] * selected_y.shape[0], columns=["Project_1"])
        new_y.append(pd.concat([selected_y, project_df], axis=1))

    x_to_return = pd.concat(new_x)
    y_to_return = pd.concat(new_y)

    return x_to_return, y_to_return


# class MicrobiomeSTUNTDataCreator(object):
#     # Taken from https://github.com/jaehyun513/STUNT/blob/main/data/income.py 
#     def __init__(self, unlabeled_x, train_or_val, n_shot,  n_query, num_way=2, min_col_ratio=0.4, max_col_ratio=0.7):
#         super().__init__()
#         self.num_classes = 2
#         self.unlabeled_x = unlabeled_x
#         self.train_or_val = train_or_val
#         self.n_shot = n_shot
#         self.n_query = n_query
#         self.num_way = num_way
#         self.min_col_ratio = min_col_ratio
#         self.max_col_ratio = max_col_ratio
#         self.rng = np.random.RandomState(RANDOM_SEED)

#     def __next__(self):
#         return self.get_batch()

#     def __iter__(self):
#         return self

#     def get_batch(self):
#         # xs, ys, xq, yq = [], [], [], []
        
#         # for _ in range(1):
#         support_set = []
#         query_set = []
#         support_sety = []
#         query_sety = []

#             # if self.source == 'val':

#             #     classes = np.random.choice(class_list, num_way, replace = False)
#             #     support_idx = []
#             #     query_idx = []
#             #     for k in classes:
#             #         k_idx = np.where(y == k)[0]
#             #         permutation = np.random.permutation(len(k_idx))
#             #         k_idx = k_idx[permutation]
#             #         support_idx.append(k_idx[:num_val_shot])
#             #         query_idx.append(k_idx[num_val_shot:num_val_shot+30])
#             #     support_idx = np.concatenate(support_idx)
#             #     query_idx = np.concatenate(query_idx)
                
#             #     support_x = x[support_idx]
#             #     query_x = x[query_idx]
#             #     s_y = y[support_idx]
#             #     q_y = y[query_idx]
#             #     support_y = copy.deepcopy(s_y)
#             #     query_y = copy.deepcopy(q_y)

#             #     i = 0
#             #     for k in classes:
#             #         support_y[s_y == k] = i
#             #         query_y[q_y == k] = i
#             #         i+=1

#             #     support_set.append(support_x)
#             #     support_sety.append(support_y)
#             #     query_set.append(query_x)
#             #     query_sety.append(query_y)

#             # elif self.source == 'train':
#         tmp_x = copy.deepcopy(self.unlabeled_x)
#         min_count = 0
#         while min_count < (self.n_shot + self.n_query):
#             min_col = int(self.unlabeled_x.shape[1] * self.min_col_ratio)
#             max_col = int(self.unlabeled_x.shape[1] * self.max_col_ratio)
#             col = np.random.choice(range(min_col, max_col), 1, replace = False)[0]
#             task_idx = np.random.choice([i for i in range(self.unlabeled_x.shape[1])], col, replace = False)
#             selected_columns_to_get_labels = np.ascontiguousarray(self.unlabeled_x[:, task_idx], dtype = np.float32)

#             kmeans = KMeans(self.num_way, max_iter=20, n_init=1)
#             kmeans.fit(selected_columns_to_get_labels)
#             y = kmeans.labels_

#             class_list, counts = np.unique(y, return_counts = True)
#             min_count = min(counts)
            
#         num_to_permute = self.unlabeled_x.shape[0]
#         for t_idx in task_idx:
#             rand_perm = np.random.permutation(num_to_permute)
#             tmp_x[:, t_idx] = tmp_x[:, t_idx][rand_perm]

#         # classes = np.random.choice(class_list, self.num_way, replace = False)
            
#         support_idx = []
#         query_idx = []
#         for k in class_list:
#             k_idx = np.where(y == k)[0]
#             permutation = np.random.permutation(len(k_idx))
#             k_idx = k_idx[permutation]
#             support_idx.append(k_idx[:self.n_shot])
#             query_idx.append(k_idx[self.n_shot:self.n_shot + self.n_query])
#         support_idx = np.concatenate(support_idx)
#         query_idx = np.concatenate(query_idx)
        
#         support_x = tmp_x[support_idx]
#         query_x = tmp_x[query_idx]
#         s_y = y[support_idx]
#         q_y = y[query_idx]
#         support_y = copy.deepcopy(s_y)
#         query_y = copy.deepcopy(q_y)

#         i = 0
#         for k in class_list:
#             support_y[s_y == k] = i
#             query_y[q_y == k] = i
#             i+=1

#         # support_set.append(support_x)
#         # support_sety.append(support_y)
#         # query_set.append(query_x)
#         # query_sety.append(query_y)

#         xs = np.reshape(
#             support_x,
#             [self.num_way * self.n_shot, self.unlabeled_x.shape[1]]
#         )
#         xq = np.reshape(
#             query_x,
#             [self.num_way * self.n_query, self.unlabeled_x.shape[1]]
#         )
#         # xs = torch.from_numpy(xs).type(torch.FloatTensor)
#         # xq = torch.from_numpy(xq).type(torch.FloatTensor)
#         # ys = torch.from_numpy(support_y).type(torch.IntTensor)
#         # yq = torch.from_numpy(query_y).type(torch.IntTensor) 

#         return xs, xq, support_y, query_y       

#             # xs_k = np.concatenate(support_set, 0)
#             # xq_k = np.concatenate(query_set, 0)
#             # ys_k = np.concatenate(support_sety, 0)
#             # yq_k = np.concatenate(query_sety, 0)

#             # xs.append(xs_k)
#             # xq.append(xq_k)
#             # ys.append(ys_k)
#             # yq.append(yq_k)

#         # xs, ys = np.stack(xs, 0), np.stack(ys, 0)
#         # xq, yq = np.stack(xq, 0), np.stack(yq, 0)            

#         # if self.source == 'val':
#         #     xs = np.reshape(
#         #         xs,
#         #         [self.tasks_per_batch, num_way * num_val_shot, self.tabular_size])
#         # else:
#         #     xs = np.reshape(
#         #         xs,
#         #         [self.tasks_per_batch, num_way * self.n_shot, self.tabular_size])

#         # if self.source == 'val':
#         #     xq = np.reshape(
#         #         xq,
#         #         [self.tasks_per_batch, num_way * 30, self.tabular_size])
#         # else:
#         #     xq = np.reshape(
#         #         xq,
#         #         [self.tasks_per_batch, num_way * self.n_query, self.tabular_size])

#         # xs = xs.astype(np.float32)
#         # xq = xq.astype(np.float32)
#         # ys = ys.astype(np.float32)
#         # yq = yq.astype(np.float32)

#         # xs = torch.from_numpy(xs).type(torch.FloatTensor)
#         # xq = torch.from_numpy(xq).type(torch.FloatTensor)

#         # ys = torch.from_numpy(ys).type(torch.LongTensor)
#         # yq = torch.from_numpy(yq).type(torch.LongTensor)         

#         # batch = {'train': (xs, ys), 'test': (xq, yq)}

#         # return batch

def process_project_group(project_group, idx, data, n_tasks=25, n_data_points=5000):
    """Process a single project group and return the results"""
    print(f"Processing project group: {project_group}")
    first_data = data.loc[idx, :]
    new_x, new_y = microbiome_stunt_data_creator(
        first_data, 
        n_tasks=n_tasks, 
        n_data_points=n_data_points, 
        train_or_val="train", 
        metadata=None, 
        project_name=project_group
    )
    return new_x, new_y


def get_dataframe_memory(df):
    """Get memory usage of a pandas DataFrame in MB"""
    return df.memory_usage(deep=True).sum() / (1024 * 1024)


if __name__ == "__main__":
    start_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # Memory in MB

    set_seed()
    data = pd.read_csv(BASE_DATA_DIR / "sun_et_al_data" / "mpa4_species_profile_preprocessed.csv", index_col=0, header=0)
    metadata = pd.read_csv(BASE_DATA_DIR / "sun_et_al_data" / "sample_group_species_preprocessed.csv", index_col=0, header=0)[["Project_1", "Group"]]

    additional_data = pd.DataFrame()
    additional_metadata = pd.DataFrame()

    by_project_grouped_metadata = metadata.groupby("Project_1")
    project_groups = [(group, idx) for group, idx in by_project_grouped_metadata.groups.items()]

    num_cpus = mp.cpu_count() - 1
    print(num_cpus)

    process_func = partial(
        process_project_group, 
        data=data, 
        n_tasks=25, 
        n_data_points=5000
    )

    # Create a pool of workers
    with mp.Pool(processes=num_cpus) as pool:
        # Map the function to the project groups
        results = pool.starmap(
            process_func, 
            [(group, idx) for group, idx in project_groups]
        )
    
    # Combine results
    additional_data = pd.concat([x for x, _ in results])
    additional_metadata = pd.concat([y for _, y in results])
    
    # Save results
    additional_data.to_csv(BASE_DATA_DIR / "sun_et_al_data" / "stunt_large" / "stunt_mpa4_species_profile_preprocessed.csv")
    additional_metadata.to_csv(BASE_DATA_DIR / "sun_et_al_data" / "stunt_large" / "stunt_sample_group_species_preprocessed.csv")

    # Print memory usage statistics
    process = psutil.Process(os.getpid())
    end_memory = process.memory_info().rss / (1024 * 1024)  # Memory in MB

    print("\n===== Process Memory Usage Statistics =====")
    print(f"Initial process memory: {start_memory:.2f} MB")
    print(f"Final process memory: {end_memory:.2f} MB")
    print(f"Memory increase: {end_memory - start_memory:.2f} MB")
    
    # Get system-wide memory info
    system_memory = psutil.virtual_memory()
    print("\n===== System Memory =====")
    print(f"Total: {system_memory.total / (1024 * 1024 * 1024):.2f} GB")
    print(f"Available: {system_memory.available / (1024 * 1024 * 1024):.2f} GB")
    print(f"Used: {(system_memory.total - system_memory.available) / (1024 * 1024 * 1024):.2f} GB")

    # Get system-wide memory info
    system_memory = psutil.virtual_memory()
    print(f"\nSystem memory: {system_memory.total / (1024 * 1024 * 1024):.2f} GB total, "
          f"{system_memory.available / (1024 * 1024 * 1024):.2f} GB available")
    
    try:
        if hasattr(process.memory_info(), 'peak_wset'):  # Windows
            peak = process.memory_info().peak_wset / (1024 * 1024)
        elif hasattr(process.memory_info(), 'peak_rss'):  # Linux
            peak = process.memory_info().peak_rss / (1024 * 1024)
        else:
            peak = None
        
        if peak is not None:
            print(f"Peak memory used by this process: {peak:.2f} MB")
    except:
        print("Could not determine peak memory usage")

    # for project_group, idx in by_project_grouped_metadata.groups.items():
    #     print("processing project group: ", project_group)
    #     first_data = data.loc[idx, :]
    #     first_metadata = metadata.loc[idx, :]
    #     new_x, new_y = microbiome_stunt_data_creator(first_data, n_tasks=25, n_data_points=5000, train_or_val="train", metadata=None, project_name=project_group)
    #     additional_data = pd.concat([additional_data, new_x])
    #     additional_metadata = pd.concat([additional_metadata, new_y])
    
    # additional_data.to_csv(BASE_DATA_DIR / "sun_et_al_data" / "stunt_large" / "stunt_mpa4_species_profile_preprocessed.csv")
    # additional_metadata.to_csv(BASE_DATA_DIR / "sun_et_al_data" / "stunt_large" / "stunt_sample_group_species_preprocessed.csv")

    # data = pd.read_csv(BASE_DATA_DIR / "sun_et_al_data" / "stunt" / "stunt_mpa4_species_profile_preprocessed.csv", index_col=0, header=0)
    # metadata = pd.read_csv(BASE_DATA_DIR / "sun_et_al_data" / "stunt" / "stunt_sample_group_species_preprocessed.csv", index_col=0, header=0)
    # metadata = metadata.map(lambda x: x.split("__")[0])
    # print(metadata.value_counts("Project_1"))
    # print(len(metadata.value_counts("Project_1")))
    # print(data)
    # print(data.shape)
    # print(metadata.shape)