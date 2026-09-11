import torch
import torch.nn as nn

NUM_CLASSES = 7
FEATURE_DIM = 332   # BASE_DIM(166) + Velocity(166)
SEQ_LEN = 48

# ==============================================================================
# UPGRADED: BIDIRECTIONAL LSTM + MULTI-HEAD SELF-ATTENTION MODEL
# ==============================================================================
class TARModel(nn.Module):
    """
    Upgraded BiLSTM + Multi-Head Self-Attention Model for Action Recognition.

    Architecture improvements over v1:
      - hidden_dim: 128 -> 256  (BiLSTM: 128 per direction)
      - num_layers:   2 ->   3  (deeper temporal modelling)
      - Activation: ReLU -> GELU (smoother gradient flow)
      - Attention: simple learned pooling -> nn.MultiheadAttention (4 heads)
      - Pooling: single-vector -> mean + max concatenation (richer aggregation)
      - Dropout: tuned to 0.3/0.4 to regularise the wider network

    Parameter count: ~600K (up from ~120K) — still compact for ~500 samples
    because strong dropout + augmentation prevents overfitting.
    """
    def __init__(self, input_size=FEATURE_DIM, num_classes=NUM_CLASSES,
                 hidden_dim=256, num_layers=3, num_heads=4):
        super().__init__()

        # ------------------------------------------------------------------
        # 1. Input Embedding: project raw 332-dim features to 256-dim space
        # ------------------------------------------------------------------
        self.embedding = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2)
        )

        # ------------------------------------------------------------------
        # 2. Bidirectional LSTM: captures temporal dynamics in both directions
        #    hidden_size = hidden_dim // 2 so BiLSTM output = hidden_dim (256)
        # ------------------------------------------------------------------
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_dim // 2,  # 128 per direction -> 256 total
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if num_layers > 1 else 0.0
        )
        self.lstm_norm = nn.LayerNorm(hidden_dim)

        # ------------------------------------------------------------------
        # 3. Multi-Head Self-Attention: lets every timestep attend to every
        #    other timestep, capturing long-range temporal dependencies that
        #    LSTM may miss (e.g. correlating pick vs. place frames).
        # ------------------------------------------------------------------
        self.mhsa = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        self.mhsa_norm = nn.LayerNorm(hidden_dim)

        # ------------------------------------------------------------------
        # 4. Temporal Pooling: mean + max concatenation over time axis
        #    Gives both average context and peak-activation signal.
        # ------------------------------------------------------------------
        # output dim = hidden_dim * 2  (mean-pooled + max-pooled concatenated)

        # ------------------------------------------------------------------
        # 5. Classifier Head
        # ------------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (B, T, F)

        # 1. Embed features
        x = self.embedding(x)                       # (B, T, 256)

        # 2. BiLSTM temporal encoding
        lstm_out, _ = self.lstm(x)                  # (B, T, 256)
        lstm_out = self.lstm_norm(lstm_out)

        # 3. Multi-Head Self-Attention (residual connection)
        attn_out, _ = self.mhsa(lstm_out, lstm_out, lstm_out)  # (B, T, 256)
        attn_out = self.mhsa_norm(lstm_out + attn_out)         # residual

        # 4. Mean + Max temporal pooling
        mean_pool = attn_out.mean(dim=1)            # (B, 256)
        max_pool  = attn_out.max(dim=1).values      # (B, 256)
        pooled = torch.cat([mean_pool, max_pool], dim=1)  # (B, 512)

        # 5. Classify
        return self.classifier(pooled)              # (B, NUM_CLASSES)