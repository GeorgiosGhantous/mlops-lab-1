"""
Train a ResNet18 classifier on the Food-11 dataset, with MLflow tracking.

Usage:
    uv run python ./src/food11/train.py --dataset mini --epochs 5 --lr 0.001 --batch-size 32
"""

import argparse
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Train a ResNet18 on Food-11")
    parser.add_argument(
        "--dataset",
        choices=["processed", "mini"],
        default="mini",
        help="Which processed dataset to use: 'mini' (fast, for dev) or 'processed' (full)",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Root directory containing food11_processed[_mini]/{training,validation,evaluation} folders",
    )
    return parser.parse_args()


def get_dataloaders(data_root: str, dataset_name: str, batch_size: int):
    folder_name = "food11_processed_mini" if dataset_name == "mini" else "food11_processed"
    root = Path(data_root) / folder_name

    # Standard ImageNet normalization since we start from ImageNet-pretrained weights
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_ds = datasets.ImageFolder(root / "training", transform=train_transform)
    val_ds = datasets.ImageFolder(root / "validation", transform=eval_transform)
    test_ds = datasets.ImageFolder(root / "evaluation", transform=eval_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, len(train_ds.classes)


def build_model(num_classes: int):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    torch.set_grad_enabled(train)
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        if train:
            optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    torch.set_grad_enabled(True)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("food11")

    train_loader, val_loader, test_loader, num_classes = get_dataloaders(
        args.data_root, args.dataset, args.batch_size
    )

    model = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    with mlflow.start_run():
        mlflow.log_params(
            {
                "dataset": args.dataset,
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "model": "resnet18",
                "num_classes": num_classes,
            }
        )

        for epoch in range(args.epochs):
            train_loss, train_acc = run_epoch(
                model, train_loader, criterion, optimizer, device, train=True
            )
            val_loss, val_acc = run_epoch(
                model, val_loader, criterion, optimizer, device, train=False
            )

            print(
                f"epoch {epoch + 1}/{args.epochs} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

        test_loss, test_acc = run_epoch(
            model, test_loader, criterion, optimizer, device, train=False
        )
        mlflow.log_metric("test_accuracy", test_acc)
        print(f"test_accuracy={test_acc:.4f}")

        # --- model logging -------------------------------------------------
        # mlflow.pytorch.log_model can pick the 'pt2' (torch.export) serialization
        # format in some environments, and that format REQUIRES an input_example
        # to trace the model graph. Passing input_example here fixes that error
        # regardless of which serialization format gets selected, and is good
        # practice anyway (it lets mlflow record the model's input/output schema).
        input_example = next(iter(test_loader))[0][:1].cpu().numpy()

        mlflow.pytorch.log_model(
            model,
            "model",
            input_example=input_example,
        )


if __name__ == "__main__":
    main()