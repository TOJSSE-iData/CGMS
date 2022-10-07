import pandas as pd
import numpy as np
import torch as th
from torch.utils.data import Dataset

from typing import List, Union, Dict


class AEDataset(Dataset):

    def __init__(self, npy_file):
        self.data = th.from_numpy(np.load(npy_file)).float()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return self.data[item]


class SynergyDataset(Dataset):

    def __init__(self, syn_fn, df_fn, cf_fn, use_folds: List, syn_thresh=30, double=False):
        super().__init__()
        samples = pd.read_csv(syn_fn)
        samples = samples[samples['fold'].isin(use_folds)]
        self.samples = samples
        self.ddc = samples[samples.columns[:3]].values
        self.reg_outs = samples[samples.columns[-2]].values
        if double:
            self.ddc = np.concatenate([self.ddc, self.ddc[:, [1, 0, 2]]], axis=0)
            self.reg_outs = np.concatenate([self.reg_outs, self.reg_outs], axis=0)
        self.reg_outs = th.from_numpy(self.reg_outs).float().reshape(-1, 1)
        self.cls_outs = th.zeros_like(self.reg_outs)
        for i in range(self.reg_outs.shape[0]):
            syn = self.reg_outs[i][0]
            if syn >= syn_thresh:
                self.cls_outs[i][0] = 1
        self.drug_feats = th.from_numpy(np.load(df_fn)).float()
        self.cell_feats = th.from_numpy(np.load(cf_fn)).float()

    def __len__(self):
        return self.ddc.shape[0]

    def __getitem__(self, item):
        d1, d2, c = self.ddc[item]
        drug1 = self.drug_feats[d1]
        drug2 = self.drug_feats[d2]
        cell = self.cell_feats[c]
        reg_out = self.reg_outs[item]
        cls_out = self.cls_outs[item]
        return drug1, drug2, cell, reg_out, cls_out


class SensitivityDataset(Dataset):

    def __init__(self, sen_fn, df_fn, cf_fn, use_folds: List, sen_thresh=50):
        super().__init__()
        samples = pd.read_csv(sen_fn, usecols=['drug_idx', 'cell_line_idx', 'ri', 'fold'])
        samples = samples[samples['fold'].isin(use_folds)]
        self.samples = samples
        self.dc = samples[samples.columns[:2]].values
        self.reg_outs = samples[samples.columns[2]].values
        self.reg_outs = th.from_numpy(self.reg_outs).float().reshape(-1, 1)
        self.cls_outs = th.zeros_like(self.reg_outs)
        for i in range(self.reg_outs.shape[0]):
            ri = self.reg_outs[i][0]
            if ri >= sen_thresh:
                self.cls_outs[i][0] = 1
        self.drug_feats = th.from_numpy(np.load(df_fn)).float()
        self.cell_feats = th.from_numpy(np.load(cf_fn)).float()

    def __len__(self):
        return self.dc.shape[0]

    def __getitem__(self, item):
        d1, c = self.dc[item]
        drug1 = self.drug_feats[d1]
        cell = self.cell_feats[c]
        reg_out = self.reg_outs[item]
        cls_out = self.cls_outs[item]
        return drug1, drug1, cell, reg_out, cls_out


class CGSenDataset(Dataset):

    def __init__(
        self,
        df_fp: str,
        d2i_fp: str,
        cf_fp: str,
        c2i_fp: str,
        sen_fp: str,
        use_folds: Union[int, List[int]]
    ):

        def read_x2idx(pth: str) -> Dict:
            df = pd.read_csv(pth, sep='\t')
            d = dict()
            for _, row in df.iterrows():
                d[row[0]] = int(row[1])
            return d

        drug2idx = read_x2idx(d2i_fp)
        cell2idx = read_x2idx(c2i_fp)
        self.n_drug = len(drug2idx)
        self.n_cell = len(cell2idx)
        self._header = ['drugs_idx', 'cell_lines_idx', 'ri']
        df_sen = pd.read_csv(sen_fp, sep=',')
        if type(use_folds) is int:
            use_folds = [use_folds]
        df_sen = df_sen[df_sen['folds'].isin(use_folds)]
        # df_sen[self._header[0]] = df_sen[self._header[0]].apply(lambda x: drug2idx[x])
        # df_sen[self._header[1]] = df_sen[self._header[1]].apply(lambda x: cell2idx[x])

        df_sen = df_sen[self._header]
        drug_feats = np.load(df_fp)
        cell_feats = np.load(cf_fp)
        self._drug2idx = drug2idx
        self._cell2idx = cell2idx
        self.drug_feats = th.from_numpy(drug_feats).float()
        self.cell_feats = th.from_numpy(cell_feats).float()
        self.samples = df_sen.values[:, :-1].astype(int)
        self.scores = th.from_numpy(df_sen.values[:, -1]).float().view(-1, 1)

    def __getitem__(self, item):
        d, c = self.samples[item]
        return self.drug_feats[[d, d]], self.cell_feats[[c]], self.scores[item]

    def __len__(self):
        return self.scores.shape[0]

    def tensor_samples(self, indices=None):
        if indices is None:
            indices = list(range(len(self)))
        d1 = self.drug_feats[[self.samples[i][0] for i in indices]]
        d2 = d1
        d = th.cat([d1, d2], dim=1)
        c = self.cell_feats[[self.samples[i][1] for i in indices]]
        y = self.scores[indices]
        return d, c, y


class CGSynDataset(Dataset):

    def __init__(self,
                 df_fp: str,
                 d2i_fp: str,
                 cf_fp: str,
                 c2i_fp: str,
                 syn_fp: str,
                 use_folds: Union[int, List[int]]):

        def read_x2idx(pth: str) -> Dict:
            df = pd.read_csv(pth, sep='\t')
            d = dict()
            for _, row in df.iterrows():
                d[row[0]] = int(row[1])
            return d

        drug2idx = read_x2idx(d2i_fp)
        cell2idx = read_x2idx(c2i_fp)
        self.n_drug = len(drug2idx)
        self.n_cell = len(cell2idx)
        self._header = ['drug_row_idx', 'drug_col_idx', 'cell_line_idx', 'synergy_loewe']
        df_syn = pd.read_csv(syn_fp, sep=',')
        if type(use_folds) is int:
            use_folds = [use_folds]
        df_syn = df_syn[df_syn['fold'].isin(use_folds)]
        # for i in [0, 1]:
        #     df_syn[self._header[i]] = df_syn[self._header[i]].apply(lambda x: drug2idx[x])
        # df_syn[self._header[2]] = df_syn[self._header[2]].apply(lambda x: cell2idx[x])

        df_syn = df_syn[self._header]
        drug_feats = np.load(df_fp)
        cell_feats = np.load(cf_fp)
        self._drug2idx = drug2idx
        self._cell2idx = cell2idx
        self.drug_feats = th.from_numpy(drug_feats).float()
        self.cell_feats = th.from_numpy(cell_feats).float()
        self.samples = df_syn.values[:, :-1].astype(int)
        self.scores = th.from_numpy(df_syn.values[:, -1]).float().view(-1, 1)

    def __getitem__(self, item):
        da, db, c = self.samples[item]
        return self.drug_feats[[da, db]], self.cell_feats[[c]], self.scores[item]

    def __len__(self):
        return self.scores.shape[0]

    def tensor_samples(self, indices=None):
        if indices is None:
            indices = list(range(len(self)))
        d1 = self.drug_feats[[self.samples[i][0] for i in indices]]
        d2 = self.drug_feats[[self.samples[i][1] for i in indices]]
        d = th.cat([d1, d2], dim=1)
        c = self.cell_feats[[self.samples[i][2] for i in indices]]
        y = self.scores[indices]
        return d, c, y


class SynDataset(Dataset):

    def __init__(self,
                 df_fp: str,
                 d2i_fp: str,
                 cf_fp: str,
                 c2i_fp: str,
                 syn_fp: str,
                 use_folds: Union[int, List[int]],
                 batch_size: int = 256):

        def read_x2idx(pth: str) -> Dict:
            df = pd.read_csv(pth, sep='\t')
            d = dict()
            for _, row in df.iterrows():
                d[row[0]] = int(row[1])
            return d

        drug2idx = read_x2idx(d2i_fp)
        cell2idx = read_x2idx(c2i_fp)
        self.n_drug = len(drug2idx)
        self.n_cell = len(cell2idx)
        self._header = ['drug_row_idx','drug_col_idx','cell_line_idx','synergy_loewe']
        df_syn = pd.read_csv(syn_fp, sep=',')
        if type(use_folds) is int:
            use_folds = [use_folds]
        df_syn = df_syn[df_syn['fold'].isin(use_folds)]
        for i in [0, 1]:
            df_syn[self._header[i]] = df_syn[self._header[i]].apply(lambda x: drug2idx[x])
        df_syn[self._header[2]] = df_syn[self._header[2]].apply(lambda x: cell2idx[x])
        self.n_valid_sample = df_syn.shape[0]

        df_syn = df_syn[self._header]
        drug_feats = np.load(df_fp)
        cell_feats = np.load(cf_fp)
        # n_to_cat = df_syn.shape[0] % batch_size
        # if n_to_cat > 0:
        #     n_to_cat = batch_size - n_to_cat
        #     df_to_cat = pd.DataFrame([(self.n_drug, self.n_drug, self.n_cell, 0)] * n_to_cat,
        #                              columns=self._header)
        #     df_syn = pd.concat([df_syn, df_to_cat])
        #     drug_feats = np.concatenate([drug_feats, np.zeros((1, drug_feats.shape[1]))], axis=0)
        #     cell_feats = np.concatenate([cell_feats, np.zeros((1, cell_feats.shape[1]))], axis=0)
        #     drug2idx.update({'Dummy Drug': self.n_drug})
        #     cell2idx.update({'Dummy Cell Line': self.n_cell})
        self._drug2idx = drug2idx
        self._cell2idx = cell2idx
        self.drug_feats = th.from_numpy(drug_feats).float()
        self.cell_feats = th.from_numpy(cell_feats).float()
        self.samples = df_syn.values[:, :-1].astype(int)
        self.scores = th.from_numpy(df_syn.values[:, -1]).float().reshape(-1, 1)

    def __getitem__(self, item):
        da, db, c = self.samples[item]
        return self.drug_feats[[da, db]], self.cell_feats[c], self.scores[item]

    def __len__(self):
        return self.samples.shape[0]


class OrderedDataset(Dataset):

    def __init__(self,
                 df_fp: str,
                 cf_fp: str,
                 syn_fp: str,
                 use_folds: Union[int, List[int]],
                 double=False,
                 test_fold=None
    ):
        self._header = ['drug_row_idx','drug_col_idx','cell_line_idx','synergy_loewe']
        df_syn = pd.read_csv(syn_fp, sep=',')
        if type(use_folds) is int:
            use_folds = [use_folds]
        df_syn = df_syn[df_syn['fold'].isin(use_folds)]
        if test_fold is not None:
            df_syn_test = df_syn.query(f'fold == {test_fold}').sample(frac=0.4)
            df_syn_trn = df_syn.query(f'fold != {test_fold}')
            df_syn = pd.concat([df_syn_trn, df_syn_test])
        self.n_valid_sample = df_syn.shape[0]

        df_syn = df_syn[self._header]

        drug_feats = np.load(df_fp)
        cell_feats = np.load(cf_fp)
        self.drug_feats = th.from_numpy(drug_feats).float()
        self.cell_feats = th.from_numpy(cell_feats).float()
        self.samples = df_syn.values[:, :-1].astype(int)
        self.scores = th.from_numpy(df_syn.values[:, -1]).float().reshape(-1, 1)
        if double:
            tmp_samples = self.samples[:, [1, 0, 2]]
            self.samples = np.vstack((self.samples, tmp_samples))
            self.scores = th.vstack([self.scores, self.scores])

    def __getitem__(self, item):
        da, db, c = self.samples[item]
        return self.drug_feats[da], self.drug_feats[db], self.cell_feats[c], self.scores[item]

    def __len__(self):
        return self.samples.shape[0]


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """

    def __init__(self, *tensors, batch_size=32, shuffle=False, device=None):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        if device:
            self.tensors = [t.to(device) for t in tensors]
        else:
            self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = th.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i + self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


class FastSynergyDataset(Dataset):

    def __init__(self, drug2id_file, cell2id_file, drug_feat_file, cell_feat_file, synergy_score_file, use_folds,
                 train=True):

        def read_x2idx(pth: str) -> Dict:
            df = pd.read_csv(pth, sep='\t')
            d = dict()
            for _, row in df.iterrows():
                d[row[0]] = int(row[1])
            return d

        self.drug2id = read_x2idx(drug2id_file)
        self.cell2id = read_x2idx(cell2id_file)
        self.drug_feat = np.load(drug_feat_file)
        self.cell_feat = np.load(cell_feat_file)
        self.samples = []
        self.raw_samples = []
        self.train = train
        valid_drugs = set(self.drug2id.keys())
        valid_cells = set(self.cell2id.keys())
        with open(synergy_score_file, 'r') as f:
            f.readline()
            for line in f:
                drug1, drug2, cellname, score, fold = line.rstrip().split('\t')
                if drug1 in valid_drugs and drug2 in valid_drugs and cellname in valid_cells:
                    if int(fold) in use_folds:
                        sample = [
                            th.from_numpy(self.drug_feat[self.drug2id[drug1]]).float(),
                            th.from_numpy(self.drug_feat[self.drug2id[drug2]]).float(),
                            th.from_numpy(self.cell_feat[self.cell2id[cellname]]).float(),
                            th.FloatTensor([float(score)]),
                        ]
                        self.samples.append(sample)
                        raw_sample = [self.drug2id[drug1], self.drug2id[drug2], self.cell2id[cellname], score]
                        self.raw_samples.append(raw_sample)
                        if train:
                            sample = [
                                th.from_numpy(self.drug_feat[self.drug2id[drug2]]).float(),
                                th.from_numpy(self.drug_feat[self.drug2id[drug1]]).float(),
                                th.from_numpy(self.cell_feat[self.cell2id[cellname]]).float(),
                                th.FloatTensor([float(score)]),
                            ]
                            self.samples.append(sample)
                            raw_sample = [self.drug2id[drug2], self.drug2id[drug1], self.cell2id[cellname], score]
                            self.raw_samples.append(raw_sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return self.samples[item]

    def drug_feat_len(self):
        return self.drug_feat.shape[-1]

    def cell_feat_len(self):
        return self.cell_feat.shape[-1]

    def tensor_samples(self, indices=None):
        if indices is None:
            indices = list(range(len(self)))
        d1 = th.cat([th.unsqueeze(self.samples[i][0], 0) for i in indices], dim=0)
        d2 = th.cat([th.unsqueeze(self.samples[i][1], 0) for i in indices], dim=0)
        c = th.cat([th.unsqueeze(self.samples[i][2], 0) for i in indices], dim=0)
        y = th.cat([th.unsqueeze(self.samples[i][3], 0) for i in indices], dim=0)
        return d1, d2, c, y
