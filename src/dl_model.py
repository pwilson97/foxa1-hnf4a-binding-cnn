"""
CNN model for 4-class motif grammar classification from 1001bp DNA sequences.

Architecture: Multi-layer CNN with dilated convolutions, inspired by BPNet.
- Layer 1: Wide conv (k=19) to capture motif-length features
- Layers 2-5: Dilated convs (d=2,4,8,16) to capture spacing/grammar
- Global average pooling → fully connected → 4 classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MotifGrammarCNN(nn.Module):
    """CNN for classifying DNA sequences into FP/HP/CB/both_pioneer.

    Args:
        seq_len: input sequence length (default 1001)
        n_classes: number of output classes (default 4)
        n_filters: number of conv filters per layer (default 128)
        dropout: dropout rate (default 0.2)
    """

    def __init__(self, seq_len=1001, n_classes=4, n_filters=128, dropout=0.2):
        super().__init__()
        self.seq_len = seq_len
        self.n_classes = n_classes

        # Layer 1: Wide convolution to detect motifs (kernel=19 ≈ typical TF motif)
        self.conv1 = nn.Conv1d(4, n_filters, kernel_size=19, padding=9)
        self.bn1 = nn.BatchNorm1d(n_filters)

        # Dilated convolution layers to capture spacing grammar
        self.dilated_blocks = nn.ModuleList()
        dilations = [2, 4, 8, 16]
        for d in dilations:
            block = nn.Sequential(
                nn.Conv1d(n_filters, n_filters, kernel_size=7, padding=3 * d,
                          dilation=d),
                nn.BatchNorm1d(n_filters),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.dilated_blocks.append(block)

        # Classifier head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(n_filters, 64)
        self.fc_drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        """Forward pass.

        Args:
            x: (batch, 4, seq_len) one-hot encoded sequences

        Returns:
            logits: (batch, n_classes) raw class scores
        """
        # Initial motif detection
        h = F.relu(self.bn1(self.conv1(x)))  # (B, n_filters, L)

        # Dilated convs with residual connections
        for block in self.dilated_blocks:
            h = h + block(h)  # residual

        # Global average pooling → classifier
        h = self.global_pool(h).squeeze(-1)  # (B, n_filters)
        h = F.relu(self.fc1(h))
        h = self.fc_drop(h)
        logits = self.fc2(h)  # (B, n_classes)

        return logits

    def get_conv1_filters(self):
        """Extract first-layer convolutional filters as numpy array.
        Shape: (n_filters, 4, kernel_size) — can be visualized as PWMs.
        """
        return self.conv1.weight.detach().cpu().numpy()


class SimpleMotifCNN(nn.Module):
    """Compact CNN optimized for CPU training on ~8K sequences.

    3 conv layers with aggressive pooling + global avg pool + FC.
    ~50K parameters, trains in ~10s/epoch on CPU.
    """

    def __init__(self, seq_len=1001, n_classes=4, dropout=0.25):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # Layer 1: motif detection (k=19 captures TF binding motifs)
            nn.Conv1d(4, 32, kernel_size=19, padding=9),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),  # 1001 → 250
            nn.Dropout(dropout),

            # Layer 2: feature combination
            nn.Conv1d(32, 64, kernel_size=11, padding=5),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),  # 250 → 62
            nn.Dropout(dropout),

            # Layer 3: higher-order grammar features
            nn.Conv1d(64, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),  # 62 → 15
            nn.Dropout(dropout),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        h = self.conv_layers(x)
        h = self.global_pool(h).squeeze(-1)
        return self.classifier(h)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Quick smoke test
    model = MotifGrammarCNN()
    print(f"MotifGrammarCNN: {count_parameters(model):,} parameters")

    x = torch.randn(4, 4, 1001)
    out = model(x)
    print(f"Input: {x.shape} → Output: {out.shape}")
    print(f"Output example: {out[0].detach().numpy()}")

    model2 = SimpleMotifCNN()
    print(f"\nSimpleMotifCNN: {count_parameters(model2):,} parameters")
    out2 = model2(x)
    print(f"Input: {x.shape} → Output: {out2.shape}")
