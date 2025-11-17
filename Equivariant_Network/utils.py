import torch
import datetime
import numpy as np

from dataclasses import dataclass, field
from torchvision import transforms

import csv
from tqdm.auto import tqdm
from eval import validation_acc


# Constants for valid options
MODEL_OPTIONS = ["resnet18", "C4resnet18", "E4C4resnet18", "D4resnet18", "E4D4resnet18"]
DATASET_OPTIONS = ["cifar10", "cifar100"]


@dataclass
class TrainConfig:
    # “time” is kept as a string like your original code
    time: str = field(
        default_factory=lambda: datetime.datetime.now().strftime("%Y%m%d%H%M")
    )

    dataset: str = "cifar10"
    model: str = "resnet18"
    batch_size: int = 128
    epochs: int = 200
    learning_rate: float = 0.05
    no_cuda: bool = False
    seed: int = 2020
    kernel: int = 3
    bias: bool = False
    reduction: float = 2.0
    groups: int = 2
    dropout: float = 0.2
    note: str = ""
    train_aug: str = "none"  # ['none', 'sup', 'simclr']
    linear: bool = False
    head: str = "linear"


class CSVLogger:
    def __init__(self, config, fieldnames, filename="log.csv"):
        self.filename = filename
        self.csv_file = open(filename, "w")

        # Write model configuration at top of csv
        writer = csv.writer(self.csv_file)
        for arg in vars(config):
            writer.writerow([arg, getattr(config, arg)])
        writer.writerow([""])

        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()

        self.csv_file.flush()

    def writerow(self, row):
        self.writer.writerow(row)
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()


class Rotation(torch.nn.Module):
    def __init__(
        self, base_transform=transforms.ToTensor(), post_transform=transforms.ToTensor()
    ):
        super().__init__()
        self.base_transform = base_transform
        self.post_transform = post_transform
        # self.rot_at_last = rot_at_last

    def forward(self, x):
        r = np.random.randint(4)
        x = self.base_transform(x)
        x = transforms.functional.rotate(x, r * 90)
        x = self.post_transform(x)
        return x, r


def get_transformations_train(aug, adversarial=None):
    def get_color_distortion(s=0.5):  # 0.5 for CIFAR10 by default
        # s is the strength of color distortion
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
        return color_distort

    normalize = transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
    )
    if aug == "none":
        pre_transform = transforms.ToTensor()
    elif aug == "crop":
        pre_transform = transforms.Compose(
            [
                transforms.RandomCrop(32),
                transforms.ToTensor(),
            ]
        )
    elif aug == "sup":
        pre_transform = transforms.Compose(
            [
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
    elif aug == "simclr":
        pre_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(p=0.5),
                get_color_distortion(s=0.5),
                transforms.ToTensor(),
            ]
        )
    elif aug == "simclr2":
        pre_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(p=0.5),
                get_color_distortion(s=1),
                transforms.ToTensor(),
            ]
        )
    train_transform = Rotation(base_transform=pre_transform, post_transform=normalize)
    base_transform = transforms.Compose([])
    base_transform.transforms.append(transforms.ToTensor())
    base_transform.transforms.append(normalize)
    if adversarial is not None:
        adversarial_trasform = transforms.Compose(
            [transforms.ToTensor(), adversarial(), normalize]
        )
    return train_transform, base_transform, adversarial_trasform


def train_model(
    *,
    config,
    train_loader,
    test_loader,
    cnn,
    criterion,
    optimizer,
    logger,
    scheduler,
    run,
    experiment_name,
    device,
):
    best_acc = 0
    best_epoch = 0

    # Training
    cnn.to(device)
    for epoch in range(config.epochs):
        xentropy_loss_avg = 0.0
        correct = 0.0
        correct_r = 0
        total = 0.0

        progress_bar = tqdm(train_loader)
        for i, (images, labels) in enumerate(progress_bar):
            progress_bar.set_description("Epoch " + str(epoch))

            images, rotations = images

            images = images.to(device)
            rotations = rotations.to(device)
            labels = labels.to(device)

            cnn.zero_grad()
            pred, pred_rot = cnn(images, rot_pred=True)

            xentropy_loss = criterion(pred, labels)
            xentropy_loss += criterion(pred_rot, rotations)
            xentropy_loss.backward()
            optimizer.step()

            xentropy_loss_avg += xentropy_loss.item()

            # Calculate running average of accuracy
            pred = torch.max(pred.data, 1)[1]
            total += labels.size(0)
            correct += (pred == labels.data).sum().item()
            correct_r += (pred_rot.argmax(dim=1) == rotations.data).sum().item()
            accuracy = correct / total
            accuracy_r = correct_r / total

            progress_bar.set_postfix(
                xentropy="%.3f" % (xentropy_loss_avg / (i + 1)),
                acc="%.3f" % accuracy,
                acc_r="%.3f" % accuracy_r,
            )

        test_acc = validation_acc(cnn, test_loader, device)
        if test_acc >= best_acc:
            best_acc = test_acc
            best_epoch = epoch
        tqdm.write(
            "test_acc: %.5f, best_acc: %.5f, best_epoch: %d"
            % (test_acc, best_acc, best_epoch)
        )
        # scheduler.step(epoch)  # Use this line for PyTorch <1.4
        scheduler.step()  # Use this line for PyTorch >=1.4

        row = {
            "epoch": str(epoch),
            "train_acc": str(accuracy),
            "test_acc": str(test_acc),
            "rot_acc": str(accuracy_r),
        }
        logger.writerow(row)
        if (epoch + 1) % 200 == 0:
            torch.save(
                cnn.state_dict(),
                f"checkpoints/{run}/" + experiment_name + "_epoch" + str(epoch) + ".pt",
            )

    torch.save(cnn.state_dict(), f"checkpoints/{run}/" + experiment_name + ".pt")
    logger.close()


def train_model_adv(
    *,
    config,
    train_loader,
    dev_loader,
    test_loader,
    cnn,
    criterion,
    optimizer,
    logger,
    scheduler,
    run,
    experiment_name,
    device,
):
    best_acc = 0
    best_epoch = 0

    # Training
    cnn.to(device)
    for epoch in range(config.epochs):
        xentropy_loss_avg = 0.0
        correct = 0.0
        correct_r = 0
        total = 0.0

        progress_bar = tqdm(train_loader)
        for i, (images, labels) in enumerate(progress_bar):
            progress_bar.set_description("Epoch " + str(epoch))

            images, rotations = images

            images = images.to(device)
            rotations = rotations.to(device)
            labels = labels.to(device)

            cnn.zero_grad()
            pred, pred_rot = cnn(images, rot_pred=True)

            xentropy_loss = criterion(pred, labels)
            xentropy_loss += criterion(pred_rot, rotations)
            xentropy_loss.backward()
            optimizer.step()

            xentropy_loss_avg += xentropy_loss.item()

            # Calculate running average of accuracy
            pred = torch.max(pred.data, 1)[1]
            total += labels.size(0)
            correct += (pred == labels.data).sum().item()
            correct_r += (pred_rot.argmax(dim=1) == rotations.data).sum().item()
            accuracy = correct / total
            accuracy_r = correct_r / total

            progress_bar.set_postfix(
                xentropy="%.3f" % (xentropy_loss_avg / (i + 1)),
                acc="%.3f" % accuracy,
                acc_r="%.3f" % accuracy_r,
            )

        test_acc = validation_acc(cnn, dev_loader, device)
        if test_acc >= best_acc:
            best_acc = test_acc
            best_epoch = epoch
        adv_acc = validation_acc(cnn, test_loader, device)
        tqdm.write(
            "test_acc: %.5f, adv_acc: %.5f, best_acc: %.5f, best_epoch: %d"
            % (test_acc, adv_acc, best_acc, best_epoch)
        )
        # scheduler.step(epoch)  # Use this line for PyTorch <1.4
        scheduler.step()  # Use this line for PyTorch >=1.4

        row = {
            "epoch": str(epoch),
            "train_acc": str(accuracy),
            "test_acc": str(test_acc),
            "adv_acc": str(adv_acc),
            "rot_acc": str(accuracy_r),
        }
        logger.writerow(row)
        if (epoch + 1) % 200 == 0:
            torch.save(
                cnn.state_dict(),
                f"checkpoints/{run}/" + experiment_name + "_epoch" + str(epoch) + ".pt",
            )

    torch.save(cnn.state_dict(), f"checkpoints/{run}/" + experiment_name + ".pt")
    logger.close()


def test_probe(cnn, probe, loader, device):
    cnn.eval()  # change model to 'eval' mode (bn uses moving mean/var).
    probe.eval()
    correct = 0.0
    total = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            z = cnn(images)
            pred = probe(z)
        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    val_acc = correct / total
    probe.train()
    return val_acc


def train_probe(
    *,
    config,
    train_loader,
    test_loader,
    cnn,
    probe,
    criterion,
    optimizer,
    logger,
    scheduler,
    experiment_name,
    device,
):
    best_acc = 0
    best_epoch = 0
    cnn.eval()
    probe.train()
    for epoch in range(config.epochs):
        xentropy_loss_avg = 0.0
        correct = 0.0
        correct_r = 0
        total = 0.0

        progress_bar = tqdm(train_loader)
        for i, (images, labels) in enumerate(progress_bar):
            progress_bar.set_description("Epoch " + str(epoch))

            images, rotations = images

            images = images.to(device)
            rotations = rotations.to(device)
            labels = labels.to(device)

            cnn.zero_grad()
            probe.zero_grad()
            z = cnn(images)
            pred = probe(z)
            xentropy_loss = criterion(pred, labels)
            xentropy_loss.backward()
            optimizer.step()

            xentropy_loss_avg += xentropy_loss.item()

            # Calculate running average of accuracy
            pred = torch.max(pred.data, 1)[1]
            total += labels.size(0)
            correct += (pred == labels.data).sum().item()
            correct_r += (z.argmax(dim=1) == rotations.data).sum().item()
            accuracy = correct / total
            accuracy_r = correct_r / total

            logger.writerow(
                {
                    "train_loss": str(xentropy_loss.item()),
                    "train_acc": str(accuracy),
                    "rot_acc": str(accuracy_r),
                }
            )

            progress_bar.set_postfix(
                xentropy="%.3f" % (xentropy_loss_avg / (i + 1)),
                acc="%.3f" % accuracy,
                acc_r="%.3f" % accuracy_r,
            )

        test_acc = test_probe(cnn, probe, test_loader, device)
        if test_acc >= best_acc:
            best_acc = test_acc
            best_epoch = epoch
        tqdm.write(
            "test_acc: %.5f, best_acc: %.5f, best_epoch: %d"
            % (test_acc, best_acc, best_epoch)
        )
        # scheduler.step(epoch)  # Use this line for PyTorch <1.4
        scheduler.step()  # Use this line for PyTorch >=1.4

        row = {
            "epoch": str(epoch),
            "train_acc": str(accuracy),
            "test_acc": str(test_acc),
            "rot_acc": str(accuracy_r),
        }
        logger.writerow(row)
    logger.close()
