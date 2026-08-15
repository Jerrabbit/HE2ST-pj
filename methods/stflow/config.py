"""STFlow 配置命名空间（字段名与官方 denoiser.py / transformer.py 一致）。"""


class SpatialConfig:
    """SpatialTransformer 配置。"""

    def __init__(self, n_genes, feature_dim, hidden_dim=256, d_edge_model=64,
                 n_layers=4, n_heads=4, dropout=0.1, attn_dropout=0.0,
                 n_neighbors=8, act='gelu'):
        self.n_genes = n_genes
        self.d_input = feature_dim
        self.d_model = hidden_dim
        self.d_edge_model = d_edge_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.n_neighbors = n_neighbors
        self.act = act


class STFlowConfig:
    """Denoiser 配置。"""

    def __init__(self, n_genes, feature_dim, hidden_dim=256, pairwise_hidden_dim=64,
                 n_layers=4, n_heads=4, dropout=0.1, attn_dropout=0.0,
                 n_neighbors=8, activation='gelu'):
        self.n_genes = n_genes
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.pairwise_hidden_dim = pairwise_hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.n_neighbors = n_neighbors
        self.activation = activation
