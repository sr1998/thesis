from re import I
import numpy as np
import pandas as pd
import torch
import os
import copy
import faiss
from src.global_vars import BASE_DATA_DIR, RANDOM_SEED


class MicrobiomeSTUNTDataCreator(object):
    # Taken from https://github.com/jaehyun513/STUNT/blob/main/data/income.py 
    def __init__(self, unlabeled_x, train_or_val, n_shot,  n_query, num_way=2, min_col_ratio=0.4, max_col_ratio=0.7):
        super().__init__()
        self.num_classes = 2
        self.unlabeled_x = unlabeled_x
        self.train_or_val = train_or_val
        self.n_shot = n_shot
        self.n_query = n_query
        self.num_way = num_way
        self.min_col_ratio = min_col_ratio
        self.max_col_ratio = max_col_ratio
        self.rng = np.random.RandomState(RANDOM_SEED)

    def __next__(self):
        return self.get_batch()

    def __iter__(self):
        return self

    def get_batch(self):
        xs, ys, xq, yq = [], [], [], []
        
        # for _ in range(1):
        support_set = []
        query_set = []
        support_sety = []
        query_sety = []

            # if self.source == 'val':

            #     classes = np.random.choice(class_list, num_way, replace = False)
            #     support_idx = []
            #     query_idx = []
            #     for k in classes:
            #         k_idx = np.where(y == k)[0]
            #         permutation = np.random.permutation(len(k_idx))
            #         k_idx = k_idx[permutation]
            #         support_idx.append(k_idx[:num_val_shot])
            #         query_idx.append(k_idx[num_val_shot:num_val_shot+30])
            #     support_idx = np.concatenate(support_idx)
            #     query_idx = np.concatenate(query_idx)
                
            #     support_x = x[support_idx]
            #     query_x = x[query_idx]
            #     s_y = y[support_idx]
            #     q_y = y[query_idx]
            #     support_y = copy.deepcopy(s_y)
            #     query_y = copy.deepcopy(q_y)

            #     i = 0
            #     for k in classes:
            #         support_y[s_y == k] = i
            #         query_y[q_y == k] = i
            #         i+=1

            #     support_set.append(support_x)
            #     support_sety.append(support_y)
            #     query_set.append(query_x)
            #     query_sety.append(query_y)

            # elif self.source == 'train':
        tmp_x = copy.deepcopy(self.unlabeled_x)
        min_count = 0
        while min_count < (self.n_shot + self.n_query):
            min_col = int(self.unlabeled_x.shape[1] * self.min_col_ratio)
            max_col = int(self.unlabeled_x.shape[1] * self.max_col_ratio)
            col = np.random.choice(range(min_col, max_col), 1, replace = False)[0]
            task_idx = np.random.choice([i for i in range(self.unlabeled_x.shape[1])], col, replace = False)
            masked_x = np.ascontiguousarray(self.unlabeled_x[:, task_idx], dtype = np.float32)
            kmeans = faiss.Kmeans(masked_x.shape[1], self.num_way, niter=20, nredo=1, verbose=False, min_points_per_centroid = self.n_shot + self.n_query, gpu=1)
            kmeans.train(masked_x)
            D, I = kmeans.index.search(masked_x, 1)
            y = I[:,0].astype(np.int32)
            class_list, counts = np.unique(y, return_counts = True)
            min_count = min(counts)
            
        num_to_permute = self.unlabeled_x.shape[0]
        for t_idx in task_idx:
            rand_perm = np.random.permutation(num_to_permute)
            tmp_x[:, t_idx] = tmp_x[:, t_idx][rand_perm]

        classes = np.random.choice(class_list, self.num_way, replace = False)
            
        support_idx = []
        query_idx = []
        for k in classes:
            k_idx = np.where(y == k)[0]
            permutation = np.random.permutation(len(k_idx))
            k_idx = k_idx[permutation]
            support_idx.append(k_idx[:self.n_shot])
            query_idx.append(k_idx[self.n_shot:self.n_shot + self.n_query])
        support_idx = np.concatenate(support_idx)
        query_idx = np.concatenate(query_idx)
        
        support_x = tmp_x[support_idx]
        query_x = tmp_x[query_idx]
        s_y = y[support_idx]
        q_y = y[query_idx]
        support_y = copy.deepcopy(s_y)
        query_y = copy.deepcopy(q_y)

        i = 0
        for k in classes:
            support_y[s_y == k] = i
            query_y[q_y == k] = i
            i+=1

        # support_set.append(support_x)
        # support_sety.append(support_y)
        # query_set.append(query_x)
        # query_sety.append(query_y)

        xs = np.reshape(
            support_x,
            [self.num_way * self.n_shot, self.unlabeled_x.shape[1]]
        )
        xq = np.reshape(
            query_x,
            [self.num_way * self.n_query, self.unlabeled_x.shape[1]]
        )
        xs = torch.from_numpy(xs).type(torch.FloatTensor)
        xq = torch.from_numpy(xq).type(torch.FloatTensor)
        ys = torch.from_numpy(support_y).type(torch.IntTensor)
        yq = torch.from_numpy(query_y).type(torch.IntTensor) 

        return xs, xq, ys, yq       

            # xs_k = np.concatenate(support_set, 0)
            # xq_k = np.concatenate(query_set, 0)
            # ys_k = np.concatenate(support_sety, 0)
            # yq_k = np.concatenate(query_sety, 0)

            # xs.append(xs_k)
            # xq.append(xq_k)
            # ys.append(ys_k)
            # yq.append(yq_k)

        # xs, ys = np.stack(xs, 0), np.stack(ys, 0)
        # xq, yq = np.stack(xq, 0), np.stack(yq, 0)            

        # if self.source == 'val':
        #     xs = np.reshape(
        #         xs,
        #         [self.tasks_per_batch, num_way * num_val_shot, self.tabular_size])
        # else:
        #     xs = np.reshape(
        #         xs,
        #         [self.tasks_per_batch, num_way * self.n_shot, self.tabular_size])

        # if self.source == 'val':
        #     xq = np.reshape(
        #         xq,
        #         [self.tasks_per_batch, num_way * 30, self.tabular_size])
        # else:
        #     xq = np.reshape(
        #         xq,
        #         [self.tasks_per_batch, num_way * self.n_query, self.tabular_size])

        # xs = xs.astype(np.float32)
        # xq = xq.astype(np.float32)
        # ys = ys.astype(np.float32)
        # yq = yq.astype(np.float32)

        # xs = torch.from_numpy(xs).type(torch.FloatTensor)
        # xq = torch.from_numpy(xq).type(torch.FloatTensor)

        # ys = torch.from_numpy(ys).type(torch.LongTensor)
        # yq = torch.from_numpy(yq).type(torch.LongTensor)         

        # batch = {'train': (xs, ys), 'test': (xq, yq)}

        # return batch

if __name__ == "__main__":
    data = pd.read_csv(BASE_DATA_DIR / "sun_et_al_data" / "mpa4_species_profile_preprocessed.csv", index_col=0, header=0)
    stunt = MicrobiomeSTUNTDataCreator(data.values, "train", n_shot=10, n_query=10)
    xs, xq, ys, yq = next(stunt)
    print(xs.shape, xq.shape, ys.shape, yq.shape)
    print(xs)
    print(ys)
    print(xq)
    print(yq)
    