import torch


def validation_acc(cnn, loader, device):
    cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.0
    total = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            pred = cnn(images)

        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    val_acc = correct / total
    cnn.train()
    return val_acc


def test(cnn, probe, loader):
    cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.0
    total = 0.0
    for images, labels in loader:
        images = images.cuda()
        labels = labels.cuda()

        with torch.no_grad():
            z = cnn(images, rot_pred=False)
            pred = probe(z)
        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    val_acc = correct / total
    return val_acc
