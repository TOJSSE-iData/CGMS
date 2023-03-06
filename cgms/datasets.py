import pandas as pd
import numpy as np
import torch as th
from torch.utils.data import Dataset

from typing import List, Union, Dict


class AEDataset(Dataset):

    def __init__(self, npy_file: str):
        self.data = th.from_numpy(np.load(npy_file)).float()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return self.data[item]


class CGSenDataset(Dataset):

    def __init__(
        self,
        df_fp: str,
        # d2i_fp: str,
        cf_fp: str,
        # c2i_fp: str,
        sen_fp: str,
        use_folds: Union[int, List[int]]
    ):

        # def read_x2idx(pth: str) -> Dict:
        #     df = pd.read_csv(pth, sep='\t')
        #     d = dict()
        #     for _, row in df.iterrows():
        #         d[row[0]] = int(row[1])
        #     return d

        # drug2idx = read_x2idx(d2i_fp)
        # cell2idx = read_x2idx(c2i_fp)
        # self.n_drug = len(drug2idx)
        # self.n_cell = len(cell2idx)
        self._header = ['drug_idx', 'cell_line_idx', 'ri']
        df_sen = pd.read_csv(sen_fp, sep=',')
        if type(use_folds) is int:
            use_folds = [use_folds]
        df_sen = df_sen[df_sen['fold'].isin(use_folds)]
        # df_sen[self._header[0]] = df_sen[self._header[0]].apply(lambda x: drug2idx[x])
        # df_sen[self._header[1]] = df_sen[self._header[1]].apply(lambda x: cell2idx[x])

        df_sen = df_sen[self._header]
        drug_feats = np.load(df_fp)
        cell_feats = np.load(cf_fp)
        # self._drug2idx = drug2idx
        # self._cell2idx = cell2idx
        self.drug_feats = th.from_numpy(drug_feats).float()
        self.cell_feats = th.from_numpy(cell_feats).float()
        self.samples = df_sen.values[:, :-1].astype(int)
        self.scores = th.from_numpy(df_sen.values[:, -1]).float().view(-1, 1)

    def __getitem__(self, item):
        d, c = self.samples[item]
        return self.drug_feats[[d, d]], self.cell_feats[c], self.scores[item]

    def __len__(self):
        return self.scores.shape[0]


class CGSynDataset(Dataset):

    def __init__(
        self,
        df_fp: str,
        # d2i_fp: str,
        cf_fp: str,
        # c2i_fp: str,
        syn_fp: str,
        use_folds: Union[int, List[int]]
    ):
        # def read_x2idx(pth: str) -> Dict[str, int]:
        #     df = pd.read_csv(pth, sep='\t')
        #     d = dict()
        #     for _, row in df.iterrows():
        #         d[row[0]] = int(row[1])
        #     return d

        # drug2idx = read_x2idx(d2i_fp)
        # cell2idx = read_x2idx(c2i_fp)

        self._header = ['drug_row_idx', 'drug_col_idx', 'cell_line_idx', 'synergy_loewe']
        df_syn = pd.read_csv(syn_fp, sep=',')
        if type(use_folds) is int:
            use_folds = [use_folds]
        df_syn = df_syn[df_syn['fold'].isin(use_folds)]
        self.keys = df_syn[['drug_row_idx', 'drug_col_idx']].reset_index(drop=True)
        # for i in [0, 1]:
        #     df_syn[self._header[i]] = df_syn[self._header[i]].apply(lambda x: drug2idx[x])
        # df_syn[self._header[2]] = df_syn[self._header[2]].apply(lambda x: cell2idx[x])

        df_syn = df_syn[self._header]
        drug_feats = np.load(df_fp)
        cell_feats = np.load(cf_fp)
        # self._drug2idx = drug2idx
        # self._cell2idx = cell2idx
        self.drug_feats = th.from_numpy(drug_feats).float()
        self.cell_feats = th.from_numpy(cell_feats).float()
        self.samples = df_syn.values[:, :-1].astype(int)
        self.scores = th.from_numpy(df_syn.values[:, -1]).float().view(-1, 1)

    def __getitem__(self, item):
        da, db, c = self.samples[item]
        return self.drug_feats[[da, db]], self.cell_feats[c], self.scores[item]

    def __len__(self):
        return self.scores.shape[0]
