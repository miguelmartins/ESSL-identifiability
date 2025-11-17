import argparse
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms
import csv
from model.resnet import resnet18, C4resnet18, E4C4resnet18, D4resnet18, E4D4resnet18
import datetime

time = datetime.datetime.now().strftime("%Y%m%d%H%M")
model_options = ["resnet18", "C4resnet18", "E4C4resnet18", "D4resnet18", "E4D4resnet18"]
dataset_options = ["cifar10", "cifar100"]

parser = argparse.ArgumentParser(description="CNN")
parser.add_argument("--dataset", "-d", default="cifar10", choices=dataset_options)
parser.add_argument("--model", "-a", default="resnet18", choices=model_options)
parser.add_argument(
    "--batch_size",
    type=int,
    default=128,
    help="input batch size for training (default: 128)",
)
parser.add_argument(
    "--epochs", type=int, default=200, help="number of epochs to train (default: 200) "
)
parser.add_argument(
    "--learning_rate", type=float, default=0.05, help="learning rate (default: 0.1)"
)
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="enables CUDA training"
)
parser.add_argument("--seed", type=int, default=2020)
parser.add_argument("--kernel", default=3, type=int, help="kernel_size")
parser.add_argument("--bias", default=False, type=bool, help="bias")
parser.add_argument("--reduction", default=2, type=float, help="reduction_ratio")
parser.add_argument("--groups", default=2, type=int, help="groups")
parser.add_argument("--dropout", default=0.2, type=float, help="dropout_rate")
parser.add_argument("--note", type=str, default="")
parser.add_argument(
    "--train_aug",
    type=str,
    default="none",
    choices=["none", "sup", "simclr"],
    help="train_aug",
)
parser.add_argument("--linear", action="store_true", default=False, help="linear")
parser.add_argument("--head", type=str, default="linear", help="head")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
cudnn.benchmark = False  # Should make training should go faster for large models
cudnn.deterministic = True
cudnn.enabled = True

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def extract_state_dict(ckpt):
    # ckpt can be a full checkpoint or a raw state_dict
    if isinstance(ckpt, dict):
        for k in ["model_state_dict", "state_dict", "model", "net", "cnn"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt  # assume it's already a state_dict


def strip_prefix(sd, prefix):
    # remove a single leading prefix if present
    if not sd:
        return sd
    if all(k.startswith(prefix) for k in sd.keys()):
        return {k[len(prefix) :]: v for k, v in sd.items()}
    return sd


def try_load(model, state):
    # try strict, then strip common wrappers and retry
    prefixes = ["module.", "model.", "cnn.", "net.", "backbone.", "encoder."]
    # 1) direct
    try:
        model.load_state_dict(state, strict=True)
        return True
    except RuntimeError:
        pass
    # 2) iteratively strip one prefix level until it fits
    sd = state
    for p in prefixes:
        sd2 = strip_prefix(sd, p)
        if sd2 is not sd:
            try:
                model.load_state_dict(sd2, strict=True)
                return True
            except RuntimeError:
                sd = sd2
                continue
    # 3) final attempt: add/remove "module." flip
    try:
        with_module = {f"module.{k}": v for k, v in sd.items()}
        model.load_state_dict(with_module, strict=True)
        return True
    except RuntimeError:
        pass
    try:
        no_module = {k.replace("module.", "", 1): v for k, v in sd.items()}
        model.load_state_dict(no_module, strict=True)
        return True
    except RuntimeError as e:
        print("Still failed to load strictly.\n", e)
        return False


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


test_id = (
    args.dataset
    + "_probe_"
    + args.model
    + "_"
    + str(args.reduction)
    + "_"
    + str(args.groups)
    + "_"
    + str(args.dropout)
    + "_"
    + "aug_"
    + args.train_aug
    + "head_"
    + args.head
    + args.note
    + time
)

print(args)


class CSVLogger:
    def __init__(self, args, fieldnames, filename="log.csv"):
        self.filename = filename
        self.csv_file = open(filename, "w")

        # Write model configuration at top of csv
        writer = csv.writer(self.csv_file)
        for arg in vars(args):
            writer.writerow([arg, getattr(args, arg)])
        writer.writerow([""])

        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()

        self.csv_file.flush()

    def writerow(self, row):
        self.writer.writerow(row)
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()


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


# Image Preprocessing
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
base_transform = transforms.Compose([])
base_transform.transforms.append(transforms.ToTensor())
base_transform.transforms.append(normalize)
if args.train_aug == "none":
    pre_transform = transforms.ToTensor()
elif args.train_aug == "sup":
    pre_transform = transforms.Compose(
        [
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
elif args.train_aug == "simclr":
    pre_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(p=0.5),
            get_color_distortion(s=0.5),
            transforms.ToTensor(),
        ]
    )
train_transform = Rotation(base_transform=pre_transform, post_transform=normalize)

test_transform = transforms.Compose([transforms.ToTensor(), AddLinfNoise(), normalize])
num_classes = 10
train_dataset = datasets.CIFAR10(
    root="./data", train=True, transform=train_transform, download=True
)

test_dataset = datasets.CIFAR10(
    root="./data", train=False, transform=test_transform, download=True
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=2,
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    pin_memory=True,
    num_workers=2,
)
args.head = "mlp"
args.model = "c4"
if args.model == "resnet18":
    cnn = resnet18(args.dropout, num_classes=num_classes, head=args.head)
else:
    cnn = C4resnet18(args.dropout, num_classes=num_classes, head=args.head)


cnn = cnn.cuda()
ckpt_path = (
    "/home/miguelmartins/Projects/Understanding-ESSL/EquivariantNetwork/checkpoints"
)
name = "cifar10_C4resnet18_2_2_0.2_aug_suphead_mlp202510290706_epoch199.pt"
raw = torch.load(f"{ckpt_path}/{name}", map_location="cuda")
print(raw)
state = extract_state_dict(raw)
ok = try_load(cnn, state)
test_id = (
    args.dataset
    + "_probe_"
    + "c4"
    + "_"
    + str(args.reduction)
    + "_"
    + str(args.groups)
    + "_"
    + str(args.dropout)
    + "_"
    + "aug_"
    + "sup"
    + "head_"
    + args.head
    + args.note
    + time
)

if not ok:
    # quick diagnostic: show a sample key from model vs ckpt to see naming mismatch
    model_key = next(iter(cnn.state_dict().keys()))
    ckpt_key = next(iter(state.keys()))
    print("Example model key: ", model_key)
    print("Example ckpt  key: ", ckpt_key)
    raise SystemExit("Failed to load: architecture/head mismatch or unusual prefixes.")

# Now wrap for multi-GPU (if needed)
if torch.cuda.device_count() > 1:
    cnn = nn.DataParallel(cnn)
cnn = cnn.cuda()
criterion = nn.CrossEntropyLoss().cuda()

if args.linear:
    linear_params = [p for n, p in cnn.named_parameters() if "projector" in n]
    cnn_optimizer = torch.optim.SGD(
        linear_params,
        lr=args.learning_rate,
        momentum=0.9,
        nesterov=True,
        weight_decay=5e-4,
    )
else:
    cnn_optimizer = torch.optim.SGD(
        cnn.parameters(),
        lr=args.learning_rate,
        momentum=0.9,
        nesterov=True,
        weight_decay=5e-4,
    )

script_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(script_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(script_dir, "logs_loss"), exist_ok=True)
os.makedirs(os.path.join(script_dir, "checkpoints"), exist_ok=True)
filename = "logs/" + test_id + ".csv"
csv_logger = CSVLogger(
    args=args,
    fieldnames=["epoch", "train_acc", "test_acc", "rot_acc"],
    filename=filename,
)
filename_1 = "logs_loss/" + test_id + ".csv"
csv_logger_1 = CSVLogger(
    args=args, fieldnames=["train_loss", "train_acc", "rot_acc"], filename=filename_1
)


def compute_param(net):
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def test(loader):
    cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.0
    total = 0.0
    for images, labels in loader:
        images = images.cuda()
        labels = labels.cuda()

        with torch.no_grad():
            pred = cnn(images)

        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    val_acc = correct / total
    return val_acc


param = compute_param(cnn)
best_acc = 0
best_epoch = 0

cnn.eval()
probe = torch.nn.Linear(10, 128).cuda()
optimizer = torch.optim.SGD(
    probe.parameters(),
    lr=args.learning_rate,
    momentum=0.9,
    nesterov=True,
    weight_decay=5e-4,
)
scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
# Training
for epoch in range(args.epochs):
    xentropy_loss_avg = 0.0
    correct = 0.0
    correct_r = 0
    total = 0.0

    progress_bar = tqdm(train_loader)
    print("Parameters of the net: {}M".format(param / (10**6)))
    for i, (images, labels) in enumerate(progress_bar):
        progress_bar.set_description("Epoch " + str(epoch))

        images, rotations = images

        images = images.cuda()
        rotations = rotations.cuda()
        labels = labels.cuda()

        cnn.zero_grad()
        z = cnn(images, rot_pred=False)
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

        csv_logger_1.writerow(
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

    test_acc = test(test_loader)
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
    csv_logger.writerow(row)
csv_logger.close()
csv_logger_1.close()

# Right
