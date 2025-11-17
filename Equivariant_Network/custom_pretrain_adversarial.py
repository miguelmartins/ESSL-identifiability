from utils import TrainConfig, CSVLogger, get_transformations_train, train_model
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms
from model.resnet import resnet18, C4resnet18, E4C4resnet18, D4resnet18, E4D4resnet18
import os
import yaml
import gc
from utils import train_model_adv


class AddLinfNoise:
    """
    Add uniform L∞-bounded noise to an image tensor (C,H,W) in [0,1].
    """

    def __init__(
        self,
        eps=8 / 255,
        p=1.0,
        clip=(0.0, 1.0),
        same_for_all_channels=False,
        generator=None,
    ):
        """
        eps: max per-pixel magnitude (in [0,1] scale). e.g., 8/255 for CIFAR-10.
        p: probability to apply the noise (use <1.0 to sometimes skip).
        clip: min/max clamp after adding noise.
        same_for_all_channels: if True, one noise map shared across channels.
        generator: optional torch.Generator for reproducibility.
        """
        self.eps = float(eps)
        self.p = float(p)
        self.clip = clip
        self.same_for_all_channels = same_for_all_channels
        self.generator = generator

    @torch.no_grad()  # remove if you want gradient through the noise sampling
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x is expected to be float tensor in [0,1], shape (C,H,W)
        if self.p < 1.0 and torch.rand((), generator=self.generator) > self.p:
            return x
        if self.same_for_all_channels:
            noise = torch.empty(1, x.shape[-2], x.shape[-1], device=x.device).uniform_(
                -self.eps, self.eps, generator=self.generator
            )
            noise = noise.expand_as(x)
        else:
            noise = torch.empty_like(x).uniform_(
                -self.eps, self.eps, generator=self.generator
            )
        x_noisy = x + noise
        return x_noisy.clamp(*self.clip)


def get_cifar_loaders(config, adversarial=None):
    train_transform, dev_transform, test_transform = get_transformations_train(
        config.train_aug, adversarial=adversarial
    )
    train_dataset = datasets.CIFAR10(
        root="./data", train=True, transform=train_transform, download=True
    )

    dev_dataset = datasets.CIFAR10(
        root="./data", train=False, transform=dev_transform, download=True
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

    dev_loader = torch.utils.data.DataLoader(
        dataset=dev_dataset,
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
    return train_loader, dev_loader, test_loader


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
        fieldnames=[
            "epoch",
            "train_loss",
            "train_acc",
            "test_acc",
            "adv_acc",
            "rot_acc",
        ],
        filename=f"{target_path}.csv",
    )
    return metric_logger, experiment_name


def run_experiment(*, device, config, config_path, run):
    num_classes = 10
    train_loader, dev_loader, test_loader = get_cifar_loaders(
        config, adversarial=AddLinfNoise
    )
    if config.model == "resnet18":
        cnn = resnet18(config.dropout, num_classes=num_classes, head=config.head)
    else:
        cnn = C4resnet18(config.dropout, num_classes=num_classes, head=config.head)
    criterion, optimizer, scheduler = get_optimization_objects(config, cnn)

    metric_logger, experiment_name = setup_logs(config, config_path, run_number=run)
    try:
        train_model_adv(
            config=config,
            train_loader=train_loader,
            dev_loader=dev_loader,
            test_loader=test_loader,
            cnn=cnn,
            criterion=criterion,
            optimizer=optimizer,
            logger=metric_logger,
            scheduler=scheduler,
            run=run,
            experiment_name=experiment_name,
            device=device,
        )
    finally:
        del train_loader
        del dev_loader
        del test_loader
        del cnn
        del criterion
        del optimizer
        del scheduler
        del metric_logger
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except:
            print("No cuda")


def main():
    configs = ["config_crop", "config_simclr", "config_simclr2"]
    for conf in configs:
        config_path = f"./configs/{conf}.yaml"
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
        run_experiment(device=device, config=config, config_path=config_path, run=0)


if __name__ == "__main__":
    main()
