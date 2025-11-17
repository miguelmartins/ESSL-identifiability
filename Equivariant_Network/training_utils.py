import os
import torch
import torch.nn as nn
from torchvision import datasets
from utils import get_transformations_train
from torch.optim.lr_scheduler import MultiStepLR
from utils import CSVLogger


def get_cifar_loaders(config, adversarial=None):
    train_transform, test_transform = get_transformations_train(
        config.train_aug, adversarial=adversarial
    )
    train_dataset = datasets.CIFAR10(
        root="./data", train=True, transform=train_transform, download=True
    )

    test_dataset = datasets.CIFAR10(
        root="./data", train=False, transform=test_transform, download=True
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )
    return train_loader, test_loader


def get_optimization_objects(config, cnn):
    criterion = nn.CrossEntropyLoss()

    if config.linear:
        linear_params = [p for n, p in cnn.named_parameters() if "projector" in n]
        cnn_optimizer = torch.optim.SGD(
            linear_params,
            lr=config.learning_rate,
            momentum=0.9,
            nesterov=True,
            weight_decay=5e-4,
        )
    else:
        cnn_optimizer = torch.optim.SGD(
            cnn.parameters(),
            lr=config.learning_rate,
            momentum=0.9,
            nesterov=True,
            weight_decay=5e-4,
        )

    scheduler = MultiStepLR(cnn_optimizer, milestones=[60, 120, 160], gamma=0.2)
    return criterion, cnn_optimizer, scheduler


def setup_logs(config, config_path, run_number=None, probe=False):
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./logs_loss", exist_ok=True)
    os.makedirs("./checkpoints", exist_ok=True)
    if run_number is not None:
        log_dir = os.path.join("./logs", str(run_number))
        os.makedirs(os.path.join(log_dir, "probe"), exist_ok=True)
        log_loss_dir = os.path.join("./logs_loss", str(run_number))
        os.makedirs(log_loss_dir, exist_ok=True)
        checkpoint_dir = os.path.join("./checkpoints", str(run_number))
        os.makedirs(checkpoint_dir, exist_ok=True)
        if probe:
            os.makedirs(os.path.join(log_dir, "probe"), exist_ok=True)
            os.makedirs(os.path.join(log_loss_dir, "probe"), exist_ok=True)
            os.makedirs(os.path.join(checkpoint_dir, "probe"), exist_ok=True)
    experiment_name = f"{os.path.basename(config_path)}_{config.time}"
    print(experiment_name)
    add_run = lambda x: x if run_number is None else os.path.join(x, str(run_number))
    add_probe = lambda x: x if not probe else os.path.join(x, "probe")
    target_dir = add_probe(add_run("./logs"))
    target_path = os.path.join(target_dir, f"{experiment_name}")
    metric_logger = CSVLogger(
        config=config,
        fieldnames=["epoch", "train_loss", "train_acc", "test_acc", "rot_acc"],
        filename=f"{target_path}.csv",
    )
    return metric_logger, experiment_name
