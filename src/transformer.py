import numpy as np
import matplotlib.pyplot as plt
import joblib

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import classification_report, confusion_matrix

from prepare_sequences import load_seq_splits

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

MAX_SEQ_LEN = 50
D_MODEL     = 64
N_HEADS     = 4
N_LAYERS    = 2
D_FF        = 128
DROPOUT     = 0.1
BATCH_SIZE  = 64
EPOCHS      = 30
LR          = 1e-3
NUM_CLASSES = 4


class TransformerClassifier(nn.Module):
    def __init__(self, num_features, d_model, num_heads, num_layers, d_ff,
                 num_classes, max_len, dropout=0.1):
        super().__init__()

        self.input_projection = nn.Linear(num_features, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classification_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, num_classes),
        )

    def forward(self, order_sequences, padding_mask):
        batch_size, seq_len, _ = order_sequences.shape
        position_indices = torch.arange(seq_len, device=order_sequences.device).unsqueeze(0).expand(batch_size, seq_len)

        hidden = self.input_projection(order_sequences) + self.position_embedding(position_indices)

        # invert mask: TransformerEncoder expects True = IGNORE
        ignore_mask = ~padding_mask
        hidden = self.encoder(hidden, src_key_padding_mask=ignore_mask)

        # masked mean pooling (average only over non-padded positions)
        mask_expanded = padding_mask.unsqueeze(-1).float()
        pooled = (hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)

        return self.classification_head(pooled)


def train_one_epoch(model, data_loader, optimizer, loss_fn, device):
    model.train()
    running_loss, num_correct, num_samples = 0, 0, 0
    for batch_sequences, batch_masks, batch_labels in data_loader:
        batch_sequences = batch_sequences.to(device)
        batch_masks = batch_masks.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()
        predictions = model(batch_sequences, batch_masks)
        loss = loss_fn(predictions, batch_labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * len(batch_labels)
        num_correct += (predictions.argmax(1) == batch_labels).sum().item()
        num_samples += len(batch_labels)
    return running_loss / num_samples, num_correct / num_samples


@torch.no_grad()
def evaluate(model, data_loader, loss_fn, device):
    model.eval()
    running_loss, num_correct, num_samples = 0, 0, 0
    all_predicted, all_actual = [], []
    for batch_sequences, batch_masks, batch_labels in data_loader:
        batch_sequences = batch_sequences.to(device)
        batch_masks = batch_masks.to(device)
        batch_labels = batch_labels.to(device)

        predictions = model(batch_sequences, batch_masks)
        loss = loss_fn(predictions, batch_labels)

        running_loss += loss.item() * len(batch_labels)
        predicted_classes = predictions.argmax(1)
        num_correct += (predicted_classes == batch_labels).sum().item()
        num_samples += len(batch_labels)
        all_predicted.append(predicted_classes.cpu().numpy())
        all_actual.append(batch_labels.cpu().numpy())

    return (running_loss / num_samples, num_correct / num_samples,
            np.concatenate(all_predicted), np.concatenate(all_actual))


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    print("Loading sequential data …")
    (train_padded, train_mask, train_labels,
     test_padded,  test_mask,  test_labels, feature_scaler) = load_seq_splits()

    num_features = train_padded.shape[2]
    print(f"  Train: {train_padded.shape}   Test: {test_padded.shape}")
    print(f"  Features per order: {num_features}")

    train_dataset = TensorDataset(torch.FloatTensor(train_padded),
                                  torch.FloatTensor(train_mask).bool(),
                                  torch.LongTensor(train_labels))
    test_dataset = TensorDataset(torch.FloatTensor(test_padded),
                                  torch.FloatTensor(test_mask).bool(),
                                  torch.LongTensor(test_labels))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset,  batch_size=BATCH_SIZE)

    model = TransformerClassifier(
        num_features=num_features, d_model=D_MODEL, num_heads=N_HEADS,
        num_layers=N_LAYERS, d_ff=D_FF, num_classes=NUM_CLASSES,
        max_len=MAX_SEQ_LEN, dropout=DROPOUT,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")

    loss_fn   = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print("\nTraining …")
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_test_acc = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer,
                                                 loss_fn, device)
        test_loss, test_acc, _, _ = evaluate(model, test_loader, loss_fn, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), "models/transformer_best.pt")

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}  │  "
                  f"train loss {train_loss:.4f}  acc {train_acc:.4f}  │  "
                  f"test loss {test_loss:.4f}  acc {test_acc:.4f}")

    model.load_state_dict(torch.load("models/transformer_best.pt",
                                      weights_only=True))

    _, final_test_acc, predicted_labels, true_labels = evaluate(
        model, test_loader, loss_fn, device
    )

    print("\n" + "=" * 60)
    print("Classification Report")
    print("=" * 60)
    print(classification_report(true_labels, predicted_labels,
                                target_names=[f"Cluster {i}" for i in range(NUM_CLASSES)]))
    print(f"Test accuracy: {final_test_acc:.4f}")

    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    fig, ax = plt.subplots(figsize=(6, 5))
    heatmap = ax.imshow(conf_matrix, cmap="Blues")
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels([f"C{i}" for i in range(NUM_CLASSES)])
    ax.set_yticklabels([f"C{i}" for i in range(NUM_CLASSES)])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Transformer Confusion Matrix (acc={final_test_acc:.4f})")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, str(conf_matrix[i, j]), ha="center", va="center",
                    color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black")
    fig.colorbar(heatmap, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig("src/figures/transformer_confusion.png",
                dpi=150, bbox_inches="tight")
    plt.show()

    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(12, 4))
    epoch_range = range(1, EPOCHS + 1)

    loss_ax.plot(epoch_range, history["train_loss"], label="Train")
    loss_ax.plot(epoch_range, history["test_loss"],  label="Test")
    loss_ax.set_xlabel("Epoch"); loss_ax.set_ylabel("Loss")
    loss_ax.set_title("Loss Curves"); loss_ax.legend()

    acc_ax.plot(epoch_range, history["train_acc"], label="Train")
    acc_ax.plot(epoch_range, history["test_acc"],  label="Test")
    acc_ax.set_xlabel("Epoch"); acc_ax.set_ylabel("Accuracy")
    acc_ax.set_title("Accuracy Curves"); acc_ax.legend()

    plt.tight_layout()
    plt.savefig("src/figures/transformer_training_curves.png",
                dpi=150, bbox_inches="tight")
    plt.show()

    joblib.dump(feature_scaler, "models/transformer_scaler.pkl")
    print(f"\nBest model saved to models/transformer_best.pt")
    print(f"Scaler saved to models/transformer_scaler.pkl")