import torch
import torch.nn as nn

NUM_CLASSES = 7
FEATURE_DIM = 332   # BASE_DIM(166) + Velocity(166)
SEQ_LEN = 48

# ==============================================================================
# TEMPORAL ATTENTION POOLING
# ==============================================================================
class AttentionPooling(nn.Module):
    """
    Learns dynamic attention weights over all 48 timesteps to focus on the
    key frames where the action actually occurs (e.g. grasping/lifting)
    instead of arbitrarily picking only the last timestep.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (B, T, H)
        scores = self.attn(x)                   # (B, T, 1)
        weights = torch.softmax(scores, dim=1)  # (B, T, 1)
        context = torch.sum(x * weights, dim=1) # (B, H)
        return context


# ==============================================================================
# BIDIRECTIONAL LSTM + ATTENTION MODEL
# ==============================================================================
class TARModel(nn.Module):
    """
    BiLSTM + Temporal Attention Model for Action Recognition.
    - Strong temporal inductive bias (reads time forward & backward).
    - Compact parameter footprint (~120K params), preventing overfitting on small datasets.
    - Learns which frames in the 48-frame sequence carry the critical action signal.
    """
    def __init__(self, input_size=FEATURE_DIM, num_classes=NUM_CLASSES, hidden_dim=128, num_layers=2):
        super().__init__()

        self.embedding = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim // 2,  # 64 each direction -> 128 total
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )

        self.pool = AttentionPooling(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (B, T, F)
        x = self.embedding(x)          # (B, T, 128)
        lstm_out, _ = self.lstm(x)     # (B, T, 128)
        pooled = self.pool(lstm_out)   # (B, 128)
        return self.classifier(pooled)