"""
Multi-task CNN for binary FOXA1/HNF4A binding prediction.

Two sigmoid output heads sharing convolutional feature extraction layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BindingCNN(nn.Module):
    """Multi-task CNN: predict FOXA1 binding + HNF4A binding from sequence.

    Shared conv layers learn general binding features.
    Separate heads specialize per TF.
    """

    def __init__(self, seq_len=1001, n_tasks=2, n_filters=64, dropout=0.25):
        super().__init__()
        self.n_tasks = n_tasks

        # Shared feature extraction
        self.conv_layers = nn.Sequential(
            # Layer 1: motif detection
            nn.Conv1d(4, n_filters, kernel_size=19, padding=9),
            nn.BatchNorm1d(n_filters),
            nn.ReLU(),
            nn.MaxPool1d(4),  # 1001 → 250
            nn.Dropout(dropout),

            # Layer 2: motif combinations / spacing
            nn.Conv1d(n_filters, n_filters * 2, kernel_size=11, padding=5),
            nn.BatchNorm1d(n_filters * 2),
            nn.ReLU(),
            nn.MaxPool1d(4),  # 250 → 62
            nn.Dropout(dropout),

            # Layer 3: grammar features
            nn.Conv1d(n_filters * 2, n_filters * 2, kernel_size=7, padding=3),
            nn.BatchNorm1d(n_filters * 2),
            nn.ReLU(),
            nn.MaxPool1d(4),  # 62 → 15
            nn.Dropout(dropout),
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        hidden_dim = n_filters * 2  # 128

        # Separate heads for FOXA1 and HNF4A
        self.foxa1_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self.hnf4a_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x: (batch, 4, seq_len)

        Returns:
            logits: (batch, 2) — [foxa1_logit, hnf4a_logit]
        """
        h = self.conv_layers(x)
        h = self.global_pool(h).squeeze(-1)  # (B, hidden_dim)

        foxa1_logit = self.foxa1_head(h)  # (B, 1)
        hnf4a_logit = self.hnf4a_head(h)  # (B, 1)

        return torch.cat([foxa1_logit, hnf4a_logit], dim=1)  # (B, 2)

    def predict_proba(self, x):
        """Get binding probabilities."""
        logits = self.forward(x)
        return torch.sigmoid(logits)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = BindingCNN()
    print(f"BindingCNN: {count_parameters(model):,} parameters")
    x = torch.randn(4, 4, 1001)
    out = model(x)
    print(f"Input: {x.shape} → Output: {out.shape}")
    probs = torch.sigmoid(out)
    print(f"Probabilities: {probs[0].detach().numpy()}")
