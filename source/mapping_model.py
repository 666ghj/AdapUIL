import torch
import torch.nn as nn
import torch.nn.functional as F

from source.loss import MappingLossFunctions


##############################################################################
#               MAPPING MODELS
##############################################################################

class PaleMapping(nn.Module):
    def __init__(self, source_embedding, target_embedding):
        """
        Parameters
        ----------
        source_embedding: torch.Tensor or Embedding_model
            Used to get embedding vectors for nodes
        target_embedding: torch.Tensor or Embedding_model
            Used to get embedding vectors for target_nodes
        target_neighbor: dict
            dict of target_node -> target_nodes_neighbors. Used for calculate vinh_loss
        """

        super(PaleMapping, self).__init__()
        self.source_embedding = source_embedding
        self.target_embedding = target_embedding
        self.loss_fn = MappingLossFunctions()


class PaleMappingLinear(PaleMapping):
    def __init__(self, embedding_dim, source_embedding, target_embedding):
        super(PaleMappingLinear, self).__init__(source_embedding, target_embedding)
        self.maps = nn.Linear(embedding_dim, embedding_dim, bias=True)

    def loss(self, source_indices, target_indices):
        source_feats = self.source_embedding[source_indices]
        target_feats = self.target_embedding[target_indices]
        source_feats_after_mapping = self.forward(source_feats)
        batch_size = source_feats.shape[0]
        mapping_loss = self.loss_fn.loss(source_feats_after_mapping, target_feats) / batch_size
        return mapping_loss

    def forward(self, source_feats):
        ret = self.maps(source_feats)
        ret = F.normalize(ret, dim=1)
        return ret


class PaleMappingMlp(PaleMapping):
    def __init__(self, embedding_dim, source_embedding, target_embedding, activate_function='relu'):
        super(PaleMappingMlp, self).__init__(source_embedding, target_embedding)
        if activate_function == 'sigmoid':
            self.activate_function = nn.Sigmoid()
        elif activate_function == 'relu':
            self.activate_function = nn.ReLU()
        else:
            self.activate_function = nn.Tanh()
        hidden_dim = 2 * embedding_dim
        self.mlp = nn.Sequential(*[
            nn.Linear(embedding_dim, hidden_dim, bias=True),
            self.activate_function,
            nn.Linear(hidden_dim, embedding_dim, bias=True)
        ])

    def loss(self, source_indices, target_indices):
        source_feats = self.source_embedding[source_indices]
        target_feats = self.target_embedding[target_indices]
        source_feats_after_mapping = self.forward(source_feats)
        batch_size = source_feats.shape[0]
        mapping_loss = self.loss_fn.loss(source_feats_after_mapping, target_feats) / batch_size
        return mapping_loss

    def forward(self, source_feats):
        ret = self.mlp(source_feats)
        ret = F.normalize(ret, dim=1)
        return ret


class PaleMappingMlp_RWR(nn.Module):
    """将不同随机游走alpha参数进行训练，得到不同的loss值，该方法已经舍弃"""
    def __init__(self, embedding_dim, source_embedding1, target_embedding1,
                 source_embedding2, target_embedding2, activate_function='relu', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.source_embedding1 = source_embedding1
        self.target_embedding1 = target_embedding1
        self.source_embedding2 = source_embedding2
        self.target_embedding2 = target_embedding2
        self.loss_fn = MappingLossFunctions()
        if activate_function == 'sigmoid':
            self.activate_function = nn.Sigmoid()
        elif activate_function == 'relu':
            self.activate_function = nn.ReLU()
        else:
            self.activate_function = nn.Tanh()
        hidden_dim = 2 * embedding_dim
        self.mlp = nn.Sequential(*[
            nn.Linear(embedding_dim, hidden_dim, bias=True),
            self.activate_function,
            nn.Linear(hidden_dim, embedding_dim, bias=True)
        ])

    def loss(self, source_indices, target_indices):
        source_feats1 = self.source_embedding1[source_indices]
        target_feats1 = self.target_embedding1[target_indices]
        source_feats_after_mapping1 = self.forward(source_feats1)
        batch_size = source_feats1.shape[0]
        loss1 = self.loss_fn.loss(source_feats_after_mapping1, target_feats1) / batch_size

        source_feats2 = self.source_embedding2[source_indices]
        target_feats2 = self.target_embedding2[target_indices]
        source_feats_after_mapping2 = self.forward(source_feats2)
        batch_size = source_feats2.shape[0]
        loss2 = self.loss_fn.loss(source_feats_after_mapping2, target_feats2) / batch_size

        return loss1, loss2

    def forward(self, source_feats):
        ret = self.mlp(source_feats)
        ret = F.normalize(ret, dim=1)
        return ret

