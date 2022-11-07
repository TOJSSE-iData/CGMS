import torch as th
import torch.nn as nn
import torch.nn.functional as thfn
import dgl

from typing import Dict, List, Tuple, Iterable, Union

from dgl.nn import GATConv
from collections import defaultdict


def reset_linear(linear_layer: nn.Module):
    gain = nn.init.calculate_gain('relu')
    nn.init.xavier_normal_(linear_layer.weight, gain=gain)
    if linear_layer.bias is not None:
        nn.init.zeros_(linear_layer.bias)


def reset_linear_in_seq(seq_linear):
    for layer in seq_linear:
        if isinstance(layer, nn.Linear):
            reset_linear(layer)


class AutoEncoder(nn.Module):

    def __init__(self, input_dim: int, hidden_dims: List[int]):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dims[0], hidden_dims[1])
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], input_dim),
        )
        self.reconstruct_loss = nn.L1Loss(reduction='mean')

    def calc_loss(
        self,
        x: th.Tensor,
        model_outs: Union[th.Tensor, Tuple[th.Tensor]]
    ) -> th.Tensor:
        enc, dec = model_outs
        return self.reconstruct_loss(dec, x)

    def forward(self, x, ret_dec=True):
        enc = self.encoder(x)
        if ret_dec:
            dec = self.decoder(enc)
            return enc, dec
        return enc


class HANSemanticAttnLayer(nn.Module):

    def __init__(self, dim_in: int, dim_hidden: int = 256):
        super(HANSemanticAttnLayer, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.Tanh(),
            nn.Linear(dim_hidden, 1, bias=False)
        )

    def reset_parameters(self):
        reset_linear_in_seq(self.project)

    def forward(self, z: th.Tensor):
        w = self.project(z)          # N, M, 1
        beta = th.softmax(w, dim=1)  # N, M, 1
        return th.sum(beta * z, 1)   # N, H * K


class CGMSHANLayer(nn.Module):

    def __init__(
        self,
        drug_dim_in: int,
        cell_dim_in: int,
        dim_out: int,
        n_head: int = 4,
        dpr_feat: float = 0.2,
        dpr_attn: float = 0.2,
    ):
        super(CGMSHANLayer, self).__init__()
        self.dim_ins = {
            'd': drug_dim_in,
            'c': cell_dim_in
        }
        self.meta_paths = self.get_metapaths()
        self.n_head = n_head
        self.dim_out = dim_out
        # c->c could be calc without gat

        self.gat_layers = nn.ModuleDict()
        for mp in self.meta_paths[:-1]:
            if mp[0] == mp[-1]:
                in_feats = self.dim_ins[mp[0]]
            else:
                in_feats = (self.dim_ins[mp[0]], self.dim_ins[mp[-1]])
            self.gat_layers[mp] = GATConv(
                in_feats=in_feats, out_feats=dim_out, num_heads=n_head,
                feat_drop=dpr_feat, attn_drop=dpr_attn, activation=thfn.leaky_relu
            )
        self.fc_c2c = nn.Linear(self.dim_ins['c'], self.dim_out * n_head)
        self.semantic_attn = nn.ModuleDict({
            'd': HANSemanticAttnLayer(self.dim_out * self.n_head, 256),
            'c': HANSemanticAttnLayer(self.dim_out * self.n_head, 256)
        })

    @staticmethod
    def get_metapaths():
        return ('d2d', 'd2c', 'c2d', 'c2c')

    @staticmethod
    def get_graph_by_metapath(metapath: str, n_drugs: int = 2) -> dgl.DGLGraph:
        if metapath not in CGMSHANLayer.get_metapaths()[:-1]:
            raise NotImplementedError(f'unsupported metapath {metapath}')
        src_nodes = []
        dst_nodes = []
        if metapath == 'd2d':
            for i in range(n_drugs):
                src_nodes.extend([i] * n_drugs)
                dst_nodes.extend(range(n_drugs))
            g = dgl.graph((th.tensor(src_nodes), th.tensor(dst_nodes)))
            return g
        if metapath == 'd2c':
            src_nodes.extend(range(n_drugs))
            dst_nodes.extend([0] * n_drugs)
        else:
            src_nodes.extend([0] * n_drugs)
            dst_nodes.extend(range(n_drugs))
        g_dict = {(metapath[0], metapath, metapath[-1]): (th.tensor(src_nodes), th.tensor(dst_nodes))}
        g = dgl.heterograph(g_dict)
        return g

    def reset_parameters(self):
        reset_linear(self.fc_c2c)
        for k in self.gat_layers:
            self.gat_layers[k].reset_parameters()
        for k in self.semantic_attn:
            self.semantic_attn[k].reset_parameters()

    def forward(
        self,
        graphs: Dict[Tuple[str, str], dgl.DGLGraph],
        feats: Dict[str, th.Tensor]
    ):
        gat_outs = defaultdict(list)
        for mp in self.meta_paths[:-1]:
            g = graphs[mp]
            if mp[0] == mp[-1]:
                feat = feats[mp[0]]
            else:
                feat = (feats[mp[0]], feats[mp[-1]])
            out = self.gat_layers[mp](g, feat)  # N, K, H
            out = out.flatten(1)                # N, K * H
            gat_outs[mp[-1]].append(out)
        gat_outs['c'].append(thfn.leaky_relu(self.fc_c2c(feats['c'])))
        hgat_outs = {}
        for e in ['d', 'c']:
            semantic_embeddings = th.stack(gat_outs[e], dim=1)       # N, M, K * H
            hgat_outs[e] = self.semantic_attn[e](semantic_embeddings)  # N, K * H
        return hgat_outs


class CGMS(nn.Module):

    def __init__(
        self,
        # dae, cae,
        drug_dim_in: int,
        cell_dim_in: int,
        dim_hidden: int,
        n_head: int = 8,
        n_layer: int = 3,
        drop_out = 0.5,
    ) -> None:
        super(CGMS, self).__init__()
        # self.drug_ae = dae
        # self.cell_ae = cae
        self.drug_dim_in = drug_dim_in
        self.cell_dim_in = cell_dim_in
        self.dim_hidden = dim_hidden
        self.n_head = n_head
        self.hans = nn.ModuleList()
        self.hans.append(CGMSHANLayer(
            self.drug_dim_in, self.cell_dim_in, self.dim_hidden, self.n_head
        ))
        han_out_dim = self.dim_hidden * self.n_head
        for _ in range(n_layer - 1):
            self.hans.append(CGMSHANLayer(
                han_out_dim, han_out_dim, self.dim_hidden, self.n_head
            ))
        self.task_heads = nn.ModuleDict()
        self.task_heads['syn'] = nn.Sequential(
            nn.Linear(han_out_dim, han_out_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(han_out_dim // 2, 1),
        )
        self.task_heads['sen'] = nn.Sequential(
            nn.Linear(han_out_dim, han_out_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(han_out_dim // 2, 1),
        )
        self.reset_parameters()
    
    @staticmethod
    def get_metapaths() -> Tuple[str]:
        return CGMSHANLayer.get_metapaths()

    @staticmethod
    def get_graph_by_metapath(metapath: str, n_drugs: int = 2) -> dgl.DGLGraph:
        return CGMSHANLayer.get_graph_by_metapath(metapath, n_drugs)

    def reset_parameters(self) -> None:
        for layer in self.hans:
            layer.reset_parameters()
        for k in self.task_heads:
            reset_linear_in_seq(self.task_heads[k])

    def forward(
        self,
        graphs: Dict[Tuple[str, str], dgl.DGLGraph],
        feats: Dict[str, th.Tensor],
        task: str
    ):
        if task not in ['syn', 'sen']:
            raise ValueError(f'Unsupported task type: {task}')
        # dc_feats = {
        #     'd': self.drug_ae(feats['d'], False),
        #     'c': self.cell_ae(feats['c'], False)
        # }
        # feats = dc_feats
        for han in self.hans:
            feats = han(graphs, feats)
        whole_graph_emb = feats['c']
        out = self.task_heads[task](whole_graph_emb)
        return out
