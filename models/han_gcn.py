
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import BaseModel
from datasets.dataset import NodeClassificationDataset
from datasets import ACM
import numpy as np
import json
from .utils import node_classification_support_metrics


class HANGCNConv(nn.Module):
    def __init__(self, in_feats, out_feats, norm="both", weight=True, bias=True,
                 activation=None, allow_zero_in_degree=False
                 ):
        super(HANGCNConv, self).__init__()
        if norm not in ('none', 'both', 'right', 'left'):
            raise ValueError(
                'Invalid norm value. Must be either "none", "both" or "right".')

        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree

        if weight:
            self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        self._activation = activation

    def reset_parameters(self):
        if self.weight is not None:
            nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, graph, feat, weight=None, edge_weight=None):
        """
        :param graph: torch.sparse.coo_tensor
        :param feat: torch.Tensor or pair of torch.Tensor
            if a pair is given, the pair must contain two tensors, where the first tensor represents
            the source node feature and the second tensor represents the destination node feature.
            Its shape should be :math:`(N, D_{in_src})` and :math:`(N, D_{in_dst})`, where :math:`N` is the number of nodes,
            if a single tensor is given, its
            shape: (N, D), where N is the number of nodes and D is the number of input features.

        :param weight: shape: (D, D_{out})
        :param edge_weight:
        :return: shape: (N, D_{out})
        """
        if not self._allow_zero_in_degree:
            if (torch.sparse.sum(graph) == 0).any():
                raise ValueError('There are 0-in-degree nodes in the graph, '
                                 'output for those nodes will be invalid.')

        if isinstance(feat, tuple):
            feat_src, feat_dst = feat
            if self._in_feats > 0:
                if feat_src.shape[1] != self._in_feats:
                    raise ValueError(
                        'Input feature shape mismatch, expect %d but got %d' % (
                            self._in_feats, feat_src.shape[1]))
                if feat_dst.shape[1] != self._in_feats:
                    raise ValueError(
                        'Input feature shape mismatch, expect %d but got %d' % (
                            self._in_feats, feat_dst.shape[1]))
        else:
            if self._in_feats > 0 and feat.shape[1] != self._in_feats:
                raise ValueError(
                    'Input feature shape mismatch, expect %d but got %d' % (
                        self._in_feats, feat.shape[1]))
            feat_src = feat_dst = feat

        if self.weight is not None:
            if weight is not None:
                raise ValueError(
                    'External weight is provided while at the same time the module has defined its own weight.')
            else:
                weight = self.weight

        if edge_weight is not None:
            graph = graph.coleasce()
            graph_weight = graph._values() * edge_weight
            graph_weight = torch.sparse_coo_tensor(graph.indices(), graph_weight,
                                                   graph.shape)
        else:
            graph_weight = graph

        if self._norm in ['both', 'left']:
            deg = torch.sparse.sum(graph, dim=1).to_dense().clamp(min=1)
            if self._norm == 'both':
                norm = torch.pow(deg, -0.5)
            else:
                norm = 1.0 / deg
            shp = norm.shape + (1,) * (feat_src.dim() - 1)
            norm = torch.reshape(norm, shp)
            feat_src = feat_src * norm

        if self._in_feats > self._out_feats:
            if weight is not None:
                feat_src = torch.matmul(feat_src, weight)
            rst = torch.sparse.mm(graph_weight, feat_src)
        else:
            rst = torch.sparse.mm(graph_weight, feat_src)
            if weight is not None:
                rst = torch.matmul(rst, weight)

        if self._norm in ['both', 'right']:
            deg = torch.sparse.sum(graph, dim=0).to_dense().clamp(min=1)
            if self._norm == 'both':
                norm = torch.pow(deg, -0.5)
            else:
                norm = 1.0 / deg
            shp = norm.shape + (1,) * (feat_dst.dim() - 1)
            norm = torch.reshape(norm, shp)
            rst = rst * norm

        if self.bias is not None:
            rst = rst + self.bias

        if self._activation is not None:
            rst = self._activation(rst)

        return rst


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)

        return (beta * z).sum(1)  # (N, D * K)


class HANLayer(nn.Module):
    """
    HAN layer.

    Arguments
    ---------
    num_meta_paths : number of homogeneous graphs generated from the metapaths.
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability

    Inputs
    ------
    gs : list[torch.SparseTensor]
        List of graphs
    h : tensor
        Input features

    Outputs
    -------
    tensor
        The output feature
    """

    def __init__(self, num_meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(HANGCNConv(in_size, out_size * layer_num_heads,
                                          activation=F.elu,
                                          allow_zero_in_degree=True))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.num_meta_paths = num_meta_paths

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

    def forward(self, gs, h):
        semantic_embeddings = []

        for i, g in enumerate(gs):
            if self.dropout is not None:
                h = self.dropout(h)
            semantic_embeddings.append(
                self.gat_layers[i](g, h).flatten(1))  # (N, D * K)
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)  # (N, D * K)


class HAN_GCN(BaseModel):
    """
    HAN with GCN layer.
    from `Heterogeneous Graph Attention Network
    <https://arxiv.org/abs/1903.07293>`__
    We replace the GAT layer with GCN layer.
    """

    def __init__(self, config, dataset):
        super(HAN_GCN, self).__init__(config, dataset)
        assert self.dataset.hetero is True, "The dataset should be heterogeneous."
        assert isinstance(self.dataset,
                          NodeClassificationDataset), "The dataset should be NodeClassificationDataset."
        self.model_name = "HAN_GCN"
        self.dataset_adaptation()
        self.prepare_modules()
        self.to(self.device)

    def dataset_adaptation(self):
        if isinstance(self.dataset, ACM):
            self._dataset_adaptation_acm()
        else:
            raise NotImplementedError("The dataset is not implemented yet.")

    def _dataset_adaptation_acm(self):
        # read the metapaths used
        self.meta_paths = self.config['meta_paths']
        gs = [self._edges_to_metapath_adjacency(meta_path) for meta_path in
              self.meta_paths]
        # covert to torch tensor (originally scipy csr matrix)
        # use torch.sparse_coo_tensor to avoid converting to dense tensor
        tensor_gs = []
        for i in range(len(gs)):
            g = gs[i]
            g = g.tocoo()
            indices = np.vstack((g.row, g.col))
            values = g.data
            tensor_gs.append(
                torch.sparse_coo_tensor(indices, values, g.shape).to(self.device)
                .coalesce())
        self.gs = tensor_gs
        self.features = torch.tensor(self.dataset.node_features).to(self.device).float()
        self.num_meta_paths = len(self.meta_paths)

        self.train_mask = torch.from_numpy(np.array(self.dataset.labels[0])[:, 0]).type(
            torch.long).to(self.device)
        train_target = torch.from_numpy(np.array(self.dataset.labels[0])[:, 1]).type(
            torch.long).to(self.device)
        self.valid_mask = torch.from_numpy(np.array(self.dataset.labels[1])[:, 0]).type(
            torch.long).to(self.device)
        valid_target = torch.from_numpy(np.array(self.dataset.labels[1])[:, 1]).type(
            torch.long).to(self.device)
        self.test_mask = torch.from_numpy(np.array(self.dataset.labels[2])[:, 0]).type(
            torch.long).to(self.device)
        test_target = torch.from_numpy(np.array(self.dataset.labels[2])[:, 1]).type(
            torch.long).to(self.device)
        target_tensor = torch.zeros(len(self.dataset.node_features), dtype=torch.long,
                                    device=self.device)
        target_tensor[self.train_mask] = train_target
        target_tensor[self.valid_mask] = valid_target
        target_tensor[self.test_mask] = test_target
        self.labels = target_tensor

    def _edges_to_metapath_adjacency(self, meta_path):
        edges = self.dataset.edges
        edge_directions = self.dataset.edge_directions

        if len(meta_path) < 1:
            raise ValueError("Meta path should have at least one node type.")

        if len(meta_path) == 2:
            index = edge_directions['edge_types'].index([meta_path[0], meta_path[1]])
            return edges[index]

        # we need to use the multiplication of adjacency matrices to achieve the metapath
        for i in range(1, len(meta_path)):
            index = edge_directions['edge_types'].index(
                [meta_path[i - 1], meta_path[i]])
            if i == 1:
                adj = edges[index]
            else:
                adj = adj @ edges[index]

        # set the data is 1
        adj.data = np.ones_like(adj.data)

        return adj

    def prepare_modules(self):
        self.layers = nn.ModuleList()
        self.layers.append(
            HANLayer(num_meta_paths=self.num_meta_paths,
                     in_size=self.dataset.num_features,
                     out_size=self.config['hidden_units'],
                     layer_num_heads=self.config['num_heads'][0],
                     dropout=self.config['dropout']))
        for l in range(1, len(self.config['num_heads'])):
            self.layers.append(HANLayer(num_meta_paths=self.num_meta_paths,
                                        in_size=self.config['hidden_units'] *
                                                self.config['num_heads'][l - 1],
                                        out_size=self.config['hidden_units'],
                                        layer_num_heads=self.config['num_heads'][l],
                                        dropout=self.config['dropout']))
        self.predict = nn.Linear(
            self.config['hidden_units'] * self.config['num_heads'][-1],
            self.dataset.num_classes)

    def standard_input(self):
        return self.gs, self.features

    def forward(self):
        g, h = self.standard_input()

        for gnn in self.layers:
            h = gnn(g, h)

        return self.predict(h)

    def loss(self, outputs):
        return F.cross_entropy(outputs[self.train_mask], self.labels[self.train_mask])

    def custom_loss(self, outputs, mask):
        return F.cross_entropy(outputs[mask], self.labels[mask])

    def custom_forward(self, handle_fn):
        """
        Custom forward function to allow for custom handling of the input of the model.
        Some explainer and explainer metrics require modification of the input before
        passing it to the model. This function allows for that modification.
        :param handle_fn:
        :return:
        """

        g, h = handle_fn(self)

        for gnn in self.layers:
            h = gnn(g, h)

        return self.predict(h)

    def fit(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'],
                                     weight_decay=self.config['weight_decay'])
        criterion = nn.CrossEntropyLoss()

        early_stopping = 0
        loss_compared = 1e10

        for epoch in range(self.config['num_epochs']):
            self.train()
            logits = self.forward()
            loss = criterion(logits[self.train_mask], self.labels[self.train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss = loss.item()

            # evaluate by validation set
            self.eval()
            logits = self.forward()
            loss = criterion(logits[self.valid_mask], self.labels[self.valid_mask])

            metrics, results = self.evaluate(logits, self.labels, self.valid_mask)

            display_str = f"Epoch {epoch} | Train Loss: {train_loss} | Validation Loss: {loss.item()} | "
            for metric in metrics:
                display_str += f"{metric}: {results[metric]} | "
            print(display_str)

            if loss.item() < loss_compared:
                self.temp_state_dict = self.state_dict()
                early_stopping = 0
                loss_compared = loss.item()
            else:
                early_stopping += 1

            if early_stopping >= self.config['patience']:
                break

        self.load_state_dict(self.temp_state_dict)

    def evaluate(self, logits, labels, mask):
        logits_computed = logits[mask]
        labels_computed = labels[mask]
        result = {}
        for metric in self.metrics:
            result[metric] = node_classification_support_metrics[metric](
                logits_computed,
                labels_computed)
        return self.metrics, result

    def get_summary(self):
        self.summary = {}
        logits = self.forward()
        metrics, results = self.evaluate(logits, self.labels, self.test_mask)
        for metric in metrics:
            self.summary[metric] = results[metric]

        return self.summary

    def save_summary(self):
        with open(self.config['summary_path'], 'w') as f:
            json.dump(self.summary, f)
