"""STFlow 配置命名空间（字段名与官方 denoiser.py / transformer.py 一致）。

默认超参对齐官方 `stflow/model/config.py` + `app/flow/train.py`（L212-221）：
    hidden_dim=128, pairwise_hidden_dim=128, n_layers=4, n_heads=4, dropout=0.2,
    attn_dropout=0.2, n_neighbors=8, activation='swiglu'。
"""


class SpatialConfig:
    """SpatialTransformer 配置。"""

    def __init__(self, n_genes, feature_dim, hidden_dim=128, d_edge_model=128,
                 n_layers=4, n_heads=4, dropout=0.2, attn_dropout=0.2,
                 n_neighbors=8, act='swiglu'):
        assert hidden_dim % n_heads == 0, f"d_model({hidden_dim}) 必须被 n_heads({n_heads}) 整除"
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

    def __init__(self, n_genes, feature_dim, hidden_dim=128, pairwise_hidden_dim=128,
                 n_layers=4, n_heads=4, dropout=0.2, attn_dropout=0.2,
                 n_neighbors=8, activation='swiglu'):
        assert hidden_dim % n_heads == 0, f"hidden_dim({hidden_dim}) 必须被 n_heads({n_heads}) 整除"
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
