from utils import TrainConfig, CSVLogger, get_transformations_train, train_model
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms
import csv
from model.resnet import resnet18, C4resnet18, E4C4resnet18, D4resnet18, E4D4resnet18
import os
import yaml
from dataclasses import asdict
from eval import validation_acc
import gc


def get_cifar_loaders(config):
    train_transform = get_transformations_train(config.train_aug)
    train_dataset = datasets.CIFAR10(
        root="./data", train=True, transform=train_transform, download=True
    )

    test_dataset = datasets.CIFAR10(
        root="./data", train=False, transform=transforms.Compose([]), download=True
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


def setup_logs(config, config_path, run_number=None):
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./logs_loss", exist_ok=True)
    os.makedirs("./checkpoints", exist_ok=True)
    if run_number is not None:
        os.makedirs(os.path.join("./logs", str(run_number)), exist_ok=True)
        os.makedirs(os.path.join("./logs_loss", str(run_number)), exist_ok=True)
        os.makedirs(os.path.join("./checkpoints", str(run_number)), exist_ok=True)
    experiment_name = f"{os.path.basename(config_path)}_{config.time}"
    print(experiment_name)
    add_run = lambda x: x if run_number is None else os.path.join(x, str(run_number))
    target_dir = add_run("./logs")
    target_path = os.path.join(target_dir, f"{experiment_name}")
    metric_logger = CSVLogger(
        config=config,
        fieldnames=["epoch", "train_loss", "train_acc", "test_acc", "rot_acc"],
        filename=f"{target_path}.csv",
    )
    return metric_logger, experiment_name


def run_experiment(*, device, config, config_path, run):
    train_transform = get_transformations_train(config.train_aug)
    num_classes = 10
    train_loader, test_loader = get_cifar_loaders(config)
    if config.model == "resnet18":
        cnn = resnet18(config.dropout, num_classes=num_classes, head=config.head)
    else:
        cnn = C4resnet18(config.dropout, num_classes=num_classes, head=config.head)
    criterion, optimizer, scheduler = get_optimization_objects(config, cnn)

    metric_logger, experiment_name = setup_logs(config, config_path, run_number=i)
    try:
        train_model(
            config=config,
            train_loader=train_loader,
            test_loader=test_loader,
            cnn=cnn,
            criterion=criterion,
            optimizer=optimizer,
            logger=metric_logger,
            scheduler=scheduler,
            run=run,
            experiment_name=experiment_name,
        )
    finally:
        del train_loader
        del test_loader
        del cnn
        del criterion
        del optimizer
        del scheduler
        del metric_logger
        gc.collect()
        torch.cuda.empty_cache()


N_RUNS = 3


def main():
    config_path = "./configs/config.yaml"
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
        config = TrainConfig(**data)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    for run in range(N_RUNS):
        run_experiment(device=device, config=config, config_path=config_path, run=run)


if __name__ == "__main__":
    main()
