import torch as th
import torch.nn as nn
import torch.nn.functional as thfn
import dgl

from typing import Dict, List, Tuple, Iterable, Union

from dgl.nn import GATConv
from collections import defaultdict


class VAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int]
    ):
        super(VAE, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU()
        )
        self.mean_vec = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.std_vec = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], input_dim)
        )
        self.reconstruct_loss = nn.L1Loss(reduction='sum')

    @staticmethod
    def re_parameterize(mu: th.Tensor, log_var: th.Tensor):
        std = th.exp(log_var / 2)
        eps = th.randn_like(std)
        return mu + eps * std

    def calc_loss(
        self,
        x: th.Tensor,
        model_outs: Union[th.Tensor, Tuple[th.Tensor]]
    ) -> th.Tensor:
        x_hat, mu, log_var, z = model_outs
        l1 = self.reconstruct_loss(x_hat, x)
        l2 = - 0.5 * th.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return l1 + l2

    def forward(self, x):
        h = self.proj(x)
        mu = self.mean_vec(h)
        log_var = self.std_vec(h)
        z = VAE.re_parameterize(mu, log_var)
        x_hat = self.decoder(z)
        return x_hat, mu, log_var, z


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

    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return enc, dec


class MultiTaskModel(nn.Module):

    def __init__(self, drug_dim, cell_dim, hidden_dims, drop_out):
        super().__init__()
        self.drug_cell_layer = nn.Sequential(
            nn.Linear((drug_dim + cell_dim), hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU()
        )
        self.synergy_layer = nn.Sequential(
            nn.Linear(hidden_dims[1] * 2, hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(hidden_dims[2], hidden_dims[3]),
            nn.ReLU()
        )
        self.synergy_out1 = nn.Linear(hidden_dims[3], 1)  # syn reg
        self.sensitivity_layer = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[4]),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(hidden_dims[4], hidden_dims[5]),
            nn.ReLU()
        )
        self.sensitivity_out1 = nn.Linear(hidden_dims[5], 1)
        self.reg_loss = nn.MSELoss(reduction='sum')
        self.cls_loss = nn.BCEWithLogitsLoss(pos_weight=th.ones(1) * 3)

        self.reset_parameters()

    def forward(self, drug1, drug2, cell):
        d1_c = self.drug_cell_layer(th.cat([drug1, cell], dim=1))
        if drug2 is not None:
            d2_c = self.drug_cell_layer(th.cat([drug2, cell], dim=1))
            syn = self.synergy_layer(th.cat([d1_c, d2_c], dim=1))
            syn_out1 = self.synergy_out1(syn)
            return syn_out1, None
        d1_sen = self.sensitivity_layer(d1_c)
        d1_sen_out1 = self.sensitivity_out1(d1_sen)
        return None, d1_sen_out1

    def reset_parameters(self) -> None:
        def _init_linear(layer):
            nn.init.xavier_uniform_(layer.weight, 1.)
            nn.init.zeros_(layer.bias)

        def _init_seq(seq_module):
            for layer in seq_module:
                if isinstance(layer, nn.Linear):
                    _init_linear(layer)

        _init_seq(self.drug_cell_layer)
        _init_seq(self.synergy_layer)
        _init_seq(self.sensitivity_layer)
        _init_linear(self.synergy_out1)
        _init_linear(self.sensitivity_out1)

    def loss_func(self, model_outs, reg_labels):
        syn_out1, d1_sen_out1 = model_outs
        if syn_out1 is not None:
            syn_loss1 = self.reg_loss(syn_out1, reg_labels.view(-1, 1))
            syn_loss2 = 0
        else:
            syn_loss1 = syn_loss2 = 0
        if d1_sen_out1 is not None:
            d1_sen_loss1 = self.reg_loss(d1_sen_out1, reg_labels.view(-1, 1))
            d1_sen_loss2 = 0
        else:
            d1_sen_loss1 = d1_sen_loss2 = 0
        total_loss = syn_loss1 + d1_sen_loss1 + syn_loss2 + d1_sen_loss2
        return total_loss, syn_loss1, d1_sen_loss1


class HGATSemanticAttnLayer(nn.Module):

    def __init__(self, dim_in: int, dim_hidden: int = 128):
        super(HGATSemanticAttnLayer, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.Tanh(),
            nn.Linear(dim_hidden, 1, bias=False)
        )
        self.weight_init()

    def weight_init(self):
        gain = nn.init.calculate_gain('tanh')
        nn.init.xavier_uniform_(self.project[0].weight, gain)
        nn.init.xavier_uniform_(self.project[2].weight, gain)
        nn.init.zeros_(self.project[0].bias)

    def forward(self, z: th.Tensor):
        w = self.project(z)                             # N, M, 1
        beta = th.softmax(w, dim=1)                     # N, M, 1
        return th.sum(beta * z, 1)                      # N, D * K


class HGATLayer(nn.Module):

    def __init__(
            self,
            uv_types: Iterable[Iterable[str]],
            dim_ins: Dict[str, int],
            dim_out: int,
            n_head: int = 4,
            dpr_feat: float = 0.2,
            dpr_attn: float = 0.2,
    ):
        super(HGATLayer, self).__init__()
        self.uv_types = list(tuple(uv_types))
        self.v_types = set([v_type for _, v_type in self.uv_types])
        self.dim_ins = dim_ins
        self.n_head = n_head
        self.dim_out = dim_out
        self.gat_layers = nn.ModuleDict()
        for u_type, v_type in self.uv_types:
            if u_type == v_type:
                # subgraph is homo graph
                dim_in = dim_ins[u_type]
            else:
                # subgraph is unidirectional bipartite graph
                dim_in = (dim_ins[u_type], dim_ins[v_type])
            self.gat_layers['->'.join([u_type, v_type])] = GATConv(
                in_feats=dim_in, out_feats=dim_out, num_heads=n_head,
                feat_drop=dpr_feat, attn_drop=dpr_attn, activation=thfn.elu,
                allow_zero_in_degree=False, bias=True
            )
        self.semantic_attn = nn.ModuleDict()
        for v_type in self.v_types:
            self.semantic_attn[v_type] = HGATSemanticAttnLayer(self.dim_out * self.n_head, 128)

    def forward(
        self,
        graphs: Dict[Tuple[str, str], dgl.DGLGraph],
        feats: Dict[str, th.Tensor]
    ):
        gat_outs = defaultdict(list)
        for u_type, v_type in self.uv_types:
            g = graphs[(u_type, v_type)]
            gat = self.gat_layers['->'.join([u_type, v_type])]
            feat = feats[u_type] if u_type == v_type else (feats[u_type], feats[v_type])
            out = gat(g, feat).flatten(1)
            gat_outs[v_type].append(out)
        hgat_outs = dict()
        for v_type in self.v_types:
            semantic_embeddings = th.stack(gat_outs[v_type], dim=1)  # N, M, D * K
            hgat_outs[v_type] = self.semantic_attn[v_type](semantic_embeddings)  # N, D * K
        return hgat_outs

    def get_attn(
        self,
        graphs: Dict[Tuple[str, str], dgl.DGLGraph],
        feats: Dict[str, th.Tensor],
        uv_types: List[Tuple[str, str]]
    ):
        gat_outs = defaultdict(list)
        gat_attns = dict()
        for u_type, v_type in self.uv_types:
            g = graphs[(u_type, v_type)]
            gat = self.gat_layers['->'.join([u_type, v_type])]
            feat = feats[u_type] if u_type == v_type else (feats[u_type], feats[v_type])
            out, attn = gat(g, feat, get_attention=True)
            out = out.flatten(1)
            # print(u_type, v_type)
            # print(out.shape)
            # print(attn.shape)
            # if u_type == 'd':
            #     print(th.mean(th.sum(attn.view(-1, 2, 2, 8, 1), dim=2)).item())
            # else:
            #     print(th.mean(attn).item())
            gat_outs[v_type].append(out)
            gat_attns[(u_type, v_type)] = attn
        hgat_outs = dict()
        for v_type in self.v_types:
            semantic_embeddings = th.stack(gat_outs[v_type], dim=1)  # N, M, D * K
            hgat_outs[v_type] = self.semantic_attn[v_type](semantic_embeddings)  # N, D * K
        node_attns = {(u, v): gat_attns[(u, v)] for u, v in uv_types}
        return node_attns


class CGM(nn.Module):

    def __init__(
        self,
        uv_types: List[Tuple[str, str]],
        dim_ins: Dict[str, int],
        dim_hidden: int,
        dim_out: int,
        n_layer: int = 2,
        n_head: int = 8,
        dpr_feat: float = 0.,
        dpr_attn: float = 0.,
    ):
        super(CGM, self).__init__()
        self.uv_types = uv_types
        self.node_types = set([v for u, v in uv_types]) | set([u for u, v in uv_types])
        self.n_head = n_head
        self.dim_ins = dim_ins
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.hgats = nn.ModuleList()
        self.hgats.append(HGATLayer(self.uv_types, self.dim_ins, dim_hidden, n_head, dpr_feat, dpr_attn))
        for _ in range(n_layer - 2):
            self.hgats.append(HGATLayer(
                self.uv_types, {nt: dim_hidden * n_head for nt in self.node_types},
                dim_hidden, n_head, dpr_feat, dpr_attn
            ))
        self.hgats.append(HGATLayer(
            self.uv_types, {nt: dim_hidden * n_head for nt in self.node_types},
            dim_out, n_head, dpr_feat, dpr_attn
        ))
        self.dnn = nn.Sequential(
            nn.Linear(self.dim_out * self.n_head, self.dim_out),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.dim_out, self.dim_out // 2),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.dim_out // 2, 1)
        )
        self._cached_graph = None
        self._cached_coalesced_graph = {}
        self._pool_func = {
            'mean': th.mean,
            'sum': th.sum
        }
        self.weight_init()

    def weight_init(self):
        gain = nn.init.calculate_gain('leaky_relu')
        nn.init.xavier_normal_(self.dnn[0].weight, gain)
        nn.init.constant_(self.dnn[0].bias, 0)
        nn.init.xavier_normal_(self.dnn[-1].weight, gain)
        nn.init.constant_(self.dnn[-1].bias, 0)

    def forward(
            self,
            graph: dgl.DGLGraph,
            feat_dict: Dict[str, th.Tensor],
            name_cell: str,
            name_drug: str,
            n_drug_in_comb: int = 2,
            pool: str = 'mean'
    ) -> th.Tensor:
        self.cache_graph(graph)
        for gnn in self.hgats:
            feat_dict = gnn(self._cached_coalesced_graph, feat_dict)
        feat_comb = feat_dict[name_cell]
        # feat_drug = feat_dict[name_drug].view(-1, n_drug_in_comb, self.dim_out)
        # feat_drug = self._pool_func[pool](feat_drug, dim=1)
        # feat_comb = th.cat([feat_cell, feat_drug], dim=1)
        syn = self.dnn(feat_comb)
        return syn

    def cache_graph(self, graph):
        if self._cached_graph is None or self._cached_graph is not graph:
            uv_types = [(type_u, type_v) for type_u, type_e, type_v in graph.canonical_etypes]
            s1 = set(uv_types) - set(self.uv_types)
            s2 = set(self.uv_types) - set(uv_types)
            assert len(s1) == 0, f"There exists unknown meta path(s) in graph: {s1}"
            assert len(s2) == 0, f"There is missing meta path(s) in graph: {s2}"
            self._cached_graph = graph
            self._cached_coalesced_graph.clear()
            for type_u, type_e, type_v in graph.canonical_etypes:
                self._cached_coalesced_graph[(type_u, type_v)] = dgl.metapath_reachable_graph(graph, [type_e])
            for type_uv, g in self._cached_coalesced_graph.items():
                self._cached_coalesced_graph[type_uv] = g.to(graph.device)


    def get_attention(
        self,
        graph: dgl.DGLGraph,
        feat_dict: Dict[str, th.Tensor],
        uv_types: List[Tuple[str, str]]
    ) -> th.Tensor:
        self.cache_graph(graph)
        for i in range(len(self.hgats) - 1):
            gnn = self.hgats[i]
            feat_dict = gnn(self._cached_coalesced_graph, feat_dict)
        gnn = self.hgats[-1]
        node_attn_dict = gnn.get_attn(self._cached_coalesced_graph, feat_dict, uv_types)
        return node_attn_dict


class DNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(DNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, drug1_feat: th.Tensor, drug2_feat: th.Tensor, cell_feat: th.Tensor):
        feat = th.cat([drug1_feat, drug2_feat, cell_feat], 1)
        out = self.network(feat)
        return out
