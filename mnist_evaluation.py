import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from split_learning_utils import load_mnist_dataset


def make_testloader(test_batch_size=128):
    return DataLoader(
        load_mnist_dataset(train=False),
        batch_size=test_batch_size,
        shuffle=False,
    )


def evaluate_model(model, device, testloader=None, test_batch_size=128):
    model.eval()
    if testloader is None:
        testloader = make_testloader(test_batch_size)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            total_loss += F.cross_entropy(output, y, reduction="sum").item()
            total_correct += (output.argmax(dim=1) == y).sum().item()
            total_examples += y.size(0)

    return total_loss / total_examples, total_correct / total_examples


def evaluate_split_model(
    client_model,
    server_model,
    device,
    testloader=None,
    test_batch_size=128,
):
    client_model.eval()
    server_model.eval()
    if testloader is None:
        testloader = make_testloader(test_batch_size)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            output = server_model(client_model(x))
            total_loss += F.cross_entropy(output, y, reduction="sum").item()
            total_correct += (output.argmax(dim=1) == y).sum().item()
            total_examples += y.size(0)

    return total_loss / total_examples, total_correct / total_examples


def print_test_metrics(label, loss, accuracy):
    print(f"{label} test loss: {loss:.4f}")
    print(f"{label} test acc:  {accuracy * 100:.2f}%")
