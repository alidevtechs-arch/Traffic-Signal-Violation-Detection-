"""
train_fixed_camera_rule_cnn.py

Fixed-camera architecture:
1. CNN predicts traffic-light color.
2. CNN detects vehicles.
3. Rule-based logic is used later:
   if signal is red and vehicle overlaps zebra/stop-line zone -> violation.

Expected CSV:
image_path,x,y,w,h,class_id,class_name,signal_state
raw_frames/video1/frame_00000.jpg,120,300,90,70,1,violating_vehicle,red
raw_frames/video1/frame_00000.jpg,260,280,100,80,0,compliant_vehicle,red

The script treats both compliant_vehicle and violating_vehicle as vehicle boxes.
class_id is not used for training the detector; signal_state is used for signal classification.

Outputs:
- saved_models_rule/best_model.pth
- saved_models_rule/latest_model.pth
- saved_models_rule/training_log.csv
- saved_models_rule/graphs/loss_curves.png
- saved_models_rule/graphs/signal_confusion_matrix.png

Run:
python train_fixed_camera_rule_cnn.py --csv annotations.csv --epochs 30 --batch-size 8 --max-boxes 10
"""

import argparse
import csv
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


SIGNAL_MAP = {
    "red": 0,
    "yellow": 1,
    "green": 2,
    "unknown": 3,
}

SIGNAL_NAMES = {
    0: "red",
    1: "yellow",
    2: "green",
    3: "unknown",
}


class FixedCameraTrafficDataset(Dataset):
    def __init__(self, csv_path, image_size=224, max_boxes=10):
        self.csv_path = csv_path
        self.image_size = image_size
        self.max_boxes = max_boxes
        self.df = pd.read_csv(csv_path)

        required_cols = [
            "image_path", "x", "y", "w", "h", "class_id", "class_name", "signal_state"
        ]
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Missing column in CSV: {col}")

        self.image_paths = sorted(self.df["image_path"].unique().tolist())

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        original_h, original_w = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = image.astype("float32") / 255.0
        image = torch.tensor(image).permute(2, 0, 1)

        rows = self.df[self.df["image_path"] == image_path].copy()

        signal_state = str(rows["signal_state"].iloc[0]).lower().strip()
        if signal_state not in SIGNAL_MAP:
            signal_state = "unknown"
        signal_label = torch.tensor(SIGNAL_MAP[signal_state], dtype=torch.long)

        # box target: [max_boxes, 5] -> x_center, y_center, width, height, confidence
        box_target = torch.zeros((self.max_boxes, 5), dtype=torch.float32)

        rows = rows[rows["class_name"] != "no_box"]
        rows = rows.dropna(subset=["x", "y", "w", "h"])
        rows = rows.sort_values(by=["x"])

        box_index = 0
        for _, row in rows.iterrows():
            if box_index >= self.max_boxes:
                break

            x = float(row["x"])
            y = float(row["y"])
            w = float(row["w"])
            h = float(row["h"])

            x_center = (x + w / 2.0) / original_w
            y_center = (y + h / 2.0) / original_h
            width = w / original_w
            height = h / original_h

            x_center = min(max(x_center, 0.0), 1.0)
            y_center = min(max(y_center, 0.0), 1.0)
            width = min(max(width, 0.0), 1.0)
            height = min(max(height, 0.0), 1.0)

            box_target[box_index] = torch.tensor(
                [x_center, y_center, width, height, 1.0], dtype=torch.float32
            )
            box_index += 1

        return image, box_target, signal_label


class FixedCameraRuleCNN(nn.Module):
    def __init__(self, max_boxes=10, num_signal_classes=4):
        super().__init__()
        self.max_boxes = max_boxes

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
        )

        self.shared_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
        )

        self.signal_head = nn.Linear(1024, num_signal_classes)

        self.box_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, max_boxes * 5),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.features(x)
        x = self.shared_fc(x)

        signal_logits = self.signal_head(x)
        box_output = self.box_head(x)
        box_output = box_output.view(batch_size, self.max_boxes, 5)
        box_output = torch.sigmoid(box_output)

        return box_output, signal_logits


class MultiTaskRuleLoss(nn.Module):
    def __init__(self, lambda_box=5.0, lambda_conf=1.0, lambda_signal=1.0):
        super().__init__()
        self.lambda_box = lambda_box
        self.lambda_conf = lambda_conf
        self.lambda_signal = lambda_signal
        self.box_loss_fn = nn.SmoothL1Loss(reduction="none")
        self.bce = nn.BCELoss(reduction="none")
        self.signal_loss_fn = nn.CrossEntropyLoss()

    def forward(self, box_predictions, signal_logits, box_targets, signal_targets):
        pred_box = box_predictions[:, :, 0:4]
        pred_conf = box_predictions[:, :, 4]

        true_box = box_targets[:, :, 0:4]
        true_conf = box_targets[:, :, 4]

        object_mask = true_conf

        box_loss = self.box_loss_fn(pred_box, true_box).sum(dim=2)
        box_loss = (box_loss * object_mask).sum() / (object_mask.sum() + 1e-6)

        conf_loss = self.bce(pred_conf, true_conf).mean()
        signal_loss = self.signal_loss_fn(signal_logits, signal_targets)

        total_loss = (
            self.lambda_box * box_loss
            + self.lambda_conf * conf_loss
            + self.lambda_signal * signal_loss
        )
        return total_loss, box_loss.detach(), conf_loss.detach(), signal_loss.detach()


def make_confusion_matrix(num_classes, true_labels, pred_labels):
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(true_labels, pred_labels):
        matrix[int(t), int(p)] += 1
    return matrix


def plot_confusion_matrix(matrix, class_names, save_path, title):
    plt.figure(figsize=(7, 6))
    plt.imshow(matrix)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()

    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45)
    plt.yticks(ticks, class_names)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_losses(log_csv, save_path):
    df = pd.read_csv(log_csv)
    plt.figure(figsize=(10, 6))
    plt.plot(df["epoch"], df["train_total_loss"], label="Train Total Loss")
    plt.plot(df["epoch"], df["val_total_loss"], label="Validation Total Loss")
    plt.plot(df["epoch"], df["train_box_loss"], label="Train Box Loss")
    plt.plot(df["epoch"], df["train_conf_loss"], label="Train Confidence Loss")
    plt.plot(df["epoch"], df["train_signal_loss"], label="Train Signal Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    count = 0
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for images, box_targets, signal_targets in loader:
            images = images.to(device)
            box_targets = box_targets.to(device)
            signal_targets = signal_targets.to(device)

            box_predictions, signal_logits = model(images)
            loss, _, _, _ = criterion(box_predictions, signal_logits, box_targets, signal_targets)

            total_loss += loss.item()
            count += 1

            preds = torch.argmax(signal_logits, dim=1)
            true_labels.extend(signal_targets.cpu().tolist())
            pred_labels.extend(preds.cpu().tolist())

    return total_loss / max(count, 1), true_labels, pred_labels


def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = FixedCameraTrafficDataset(args.csv, args.image_size, args.max_boxes)
    total_size = len(dataset)
    train_size = int(total_size * 0.8)
    val_size = total_size - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = FixedCameraRuleCNN(max_boxes=args.max_boxes).to(device)
    criterion = MultiTaskRuleLoss(args.lambda_box, args.lambda_conf, args.lambda_signal)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)
    graph_dir = os.path.join(args.save_dir, "graphs")
    os.makedirs(graph_dir, exist_ok=True)

    log_path = os.path.join(args.save_dir, "training_log.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_total_loss", "train_box_loss", "train_conf_loss",
            "train_signal_loss", "val_total_loss"
        ])

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        train_total_sum = 0.0
        train_box_sum = 0.0
        train_conf_sum = 0.0
        train_signal_sum = 0.0

        for images, box_targets, signal_targets in train_loader:
            images = images.to(device)
            box_targets = box_targets.to(device)
            signal_targets = signal_targets.to(device)

            box_predictions, signal_logits = model(images)
            loss, box_loss, conf_loss, signal_loss = criterion(
                box_predictions, signal_logits, box_targets, signal_targets
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_total_sum += loss.item()
            train_box_sum += box_loss.item()
            train_conf_sum += conf_loss.item()
            train_signal_sum += signal_loss.item()

        avg_train_total = train_total_sum / len(train_loader)
        avg_train_box = train_box_sum / len(train_loader)
        avg_train_conf = train_conf_sum / len(train_loader)
        avg_train_signal = train_signal_sum / len(train_loader)

        val_total_loss, val_true_labels, val_pred_labels = evaluate(model, val_loader, device, criterion)

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"Train Total: {avg_train_total:.4f} "
            f"Box: {avg_train_box:.4f} "
            f"Conf: {avg_train_conf:.4f} "
            f"Signal: {avg_train_signal:.4f} "
            f"Val Total: {val_total_loss:.4f}"
        )

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1, avg_train_total, avg_train_box,
                avg_train_conf, avg_train_signal, val_total_loss
            ])

        torch.save(model.state_dict(), os.path.join(args.save_dir, "latest_model.pth"))

        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            best_path = os.path.join(args.save_dir, "best_model_3.pth")
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model: {best_path}")

    loss_plot_path = os.path.join(graph_dir, "loss_curves.png")
    plot_losses(log_path, loss_plot_path)

    _, true_labels, pred_labels = evaluate(model, val_loader, device, criterion)
    matrix = make_confusion_matrix(4, true_labels, pred_labels)
    cm_path = os.path.join(graph_dir, "signal_confusion_matrix.png")
    plot_confusion_matrix(matrix, [SIGNAL_NAMES[i] for i in range(4)], cm_path, "Traffic Signal Confusion Matrix")

    print("Training completed.")
    print(f"Training log saved to: {log_path}")
    print(f"Loss graph saved to: {loss_plot_path}")
    print(f"Signal confusion matrix saved to: {cm_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="annotations.csv")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-boxes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save-dir", type=str, default="saved_models_rule")
    parser.add_argument("--lambda-box", type=float, default=5.0)
    parser.add_argument("--lambda-conf", type=float, default=1.0)
    parser.add_argument("--lambda-signal", type=float, default=1.0)
    args = parser.parse_args()
    train_model(args)


if __name__ == "__main__":
    main()
