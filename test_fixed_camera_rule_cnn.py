import argparse
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


SIGNAL_NAMES = {
    0: "red",
    1: "yellow",
    2: "green",
    3: "unknown",
}


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


def find_images(image_dir):
    image_dir = Path(image_dir)
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    images = []
    for ext in extensions:
        images.extend(image_dir.rglob(ext))
    return sorted(images)


def preprocess_image(image_path, image_size):
    original = cv2.imread(str(image_path))
    if original is None:
        raise ValueError(f"Could not read image: {image_path}")

    original_h, original_w = original.shape[:2]
    image = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size))
    image = image.astype("float32") / 255.0
    image_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)
    return original, image_tensor, original_w, original_h


def box_overlap_ratio(box, zone):
    """
    box:  (x1, y1, x2, y2)
    zone: (x1, y1, x2, y2)
    returns intersection_area / box_area
    """
    bx1, by1, bx2, by2 = box
    zx1, zy1, zx2, zy2 = zone

    ix1 = max(bx1, zx1)
    iy1 = max(by1, zy1)
    ix2 = min(bx2, zx2)
    iy2 = min(by2, zy2)

    intersection_w = max(0, ix2 - ix1)
    intersection_h = max(0, iy2 - iy1)
    intersection_area = intersection_w * intersection_h
    box_area = max(1, (bx2 - bx1) * (by2 - by1))
    return intersection_area / box_area


def predict_image(model, image_path, image_size, conf_threshold, overlap_threshold, zone, device):
    original, image_tensor, original_w, original_h = preprocess_image(image_path, image_size)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        box_predictions, signal_logits = model(image_tensor)

    box_predictions = box_predictions[0].cpu()
    signal_probs = torch.softmax(signal_logits, dim=1)[0].cpu()

    signal_class = int(torch.argmax(signal_probs).item())
    predicted_signal = SIGNAL_NAMES[signal_class]
    signal_confidence = float(signal_probs[signal_class].item())

    results = []

    for slot_index, pred in enumerate(box_predictions):
        x_center, y_center, width, height, confidence = pred.tolist()
        if confidence < conf_threshold:
            continue

        x_center *= original_w
        y_center *= original_h
        width *= original_w
        height *= original_h

        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        x1 = max(0, min(x1, original_w - 1))
        y1 = max(0, min(y1, original_h - 1))
        x2 = max(0, min(x2, original_w - 1))
        y2 = max(0, min(y2, original_h - 1))

        overlap = box_overlap_ratio((x1, y1, x2, y2), zone)
        is_violation = predicted_signal == "red" and overlap >= overlap_threshold

        results.append({
            "slot": slot_index,
            "box": (x1, y1, x2, y2),
            "confidence": confidence,
            "overlap": overlap,
            "is_violation": is_violation,
        })

    return original, predicted_signal, signal_confidence, results


def draw_results(original, predicted_signal, signal_confidence, results, zone):
    output = original.copy()
    zx1, zy1, zx2, zy2 = zone

    cv2.rectangle(output, (zx1, zy1), (zx2, zy2), (255, 0, 0), 3)
    cv2.putText(
        output, "VIOLATION ZONE", (zx1, max(25, zy1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2
    )

    signal_text = f"Predicted Signal: {predicted_signal.upper()} ({signal_confidence:.2f})"
    cv2.rectangle(output, (0, 0), (output.shape[1], 45), (0, 0, 0), -1)
    cv2.putText(output, signal_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    for result in results:
        x1, y1, x2, y2 = result["box"]
        confidence = result["confidence"]
        overlap = result["overlap"]
        is_violation = result["is_violation"]
        slot = result["slot"]

        if is_violation:
            color = (0, 0, 255)
            label = f"VIOLATION conf={confidence:.2f} overlap={overlap:.2f}"
        else:
            color = (0, 255, 0)
            label = f"vehicle conf={confidence:.2f} overlap={overlap:.2f}"

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.putText(output, label, (x1, max(60, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.putText(output, f"slot {slot}", (x1, min(output.shape[0] - 10, y2 + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return output


def make_confusion_matrix(num_classes, true_labels, pred_labels):
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(true_labels, pred_labels):
        matrix[int(t), int(p)] += 1
    return matrix


def plot_confusion_matrix(matrix, class_names, save_path, title):
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names)
    plt.yticks(ticks, class_names)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def load_ground_truth_violation_labels(csv_path):
    """
    image_path -> 1 if image has at least one class_id == 1, else 0.
    This evaluates image-level violation presence.
    """
    if csv_path is None or not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)
    if "image_path" not in df.columns or "class_id" not in df.columns:
        return None

    label_dict = {}
    for image_path, group in df.groupby("image_path"):
        group = group.dropna(subset=["class_id"])
        has_violation = False
        for value in group["class_id"].tolist():
            try:
                if int(value) == 1:
                    has_violation = True
                    break
            except Exception:
                pass
        label_dict[os.path.relpath(image_path)] = 1 if has_violation else 0

    return label_dict


def test_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output, exist_ok=True)
    zone = tuple(args.zone)
    print(f"Using fixed violation zone: {zone}")

    model = FixedCameraRuleCNN(max_boxes=args.max_boxes).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    image_paths = find_images(args.images)
    if args.limit > 0:
        image_paths = image_paths[:args.limit]

    gt_violation_labels = load_ground_truth_violation_labels(args.csv)
    true_violation_image_labels = []
    pred_violation_image_labels = []

    print(f"Found {len(image_paths)} images.")

    for index, image_path in enumerate(image_paths, start=1):
        original, predicted_signal, signal_confidence, results = predict_image(
            model=model,
            image_path=image_path,
            image_size=args.image_size,
            conf_threshold=args.conf_threshold,
            overlap_threshold=args.overlap_threshold,
            zone=zone,
            device=device,
        )

        output = draw_results(original, predicted_signal, signal_confidence, results, zone)
        output_name = f"rule_pred_{index:05d}_{image_path.stem}.jpg"
        output_path = os.path.join(args.output, output_name)
        cv2.imwrite(output_path, output)

        pred_image_violation = 1 if any(r["is_violation"] for r in results) else 0

        if gt_violation_labels is not None:
            rel_path = os.path.relpath(image_path)
            if rel_path in gt_violation_labels:
                true_violation_image_labels.append(gt_violation_labels[rel_path])
                pred_violation_image_labels.append(pred_image_violation)

        print(
            f"[{index}/{len(image_paths)}] "
            f"signal={predicted_signal} vehicles={len(results)} "
            f"image_violation={pred_image_violation} saved={output_path}"
        )

    if len(true_violation_image_labels) > 0:
        matrix = make_confusion_matrix(2, true_violation_image_labels, pred_violation_image_labels)
        cm_path = os.path.join(args.output, "final_violation_confusion_matrix.png")
        plot_confusion_matrix(matrix, ["no_violation", "violation"], cm_path, "Final Rule-Based Violation Confusion Matrix")
        accuracy = np.trace(matrix) / max(1, matrix.sum())
        print(f"Final violation confusion matrix saved to: {cm_path}")
        print(f"Image-level rule-based violation accuracy: {accuracy:.4f}")

    print("Testing completed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="saved_models_rule/best_model.pth")
    parser.add_argument("--images", type=str, default="raw_frames")
    parser.add_argument("--csv", type=str, default="annotations.csv")
    parser.add_argument("--output", type=str, default="rule_predictions")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-boxes", type=int, default=10)
    parser.add_argument("--conf-threshold", type=float, default=0.5)
    parser.add_argument("--overlap-threshold", type=float, default=0.2)
    parser.add_argument("--zone", type=int, nargs=4, required=True, metavar=("X1", "Y1", "X2", "Y2"), help="Fixed violation zone: x1 y1 x2 y2")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    test_model(args)


if __name__ == "__main__":
    main()
