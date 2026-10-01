import copy
import math
import os
import random
from collections import OrderedDict
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, default_collate
from torchvision import datasets, transforms


SEED = 1234
_DATASET_CACHE = {}
_SPLIT_CACHE = {}
MIN_DIRICHLET_ALPHA = 1e-3
SIMULATION_TOTAL_CPUS_ENV = "SIMULATION_TOTAL_CPUS"
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
RESNET_DEPTH_PRESETS = {
    18: ("basic", (2, 2, 2, 2)),
    34: ("basic", (3, 4, 6, 3)),
    50: ("bottleneck", (3, 4, 6, 3)),
    101: ("bottleneck", (3, 4, 23, 3)),
    152: ("bottleneck", (3, 8, 36, 3)),
}
RESNET_SPLIT_POINTS = ("layer1", "layer2", "layer3", "layer4")
DEFAULT_RESNET_WIDTHS = (64, 128, 256, 512)
CNN_SPLIT_POINTS = ("conv1", "conv2")
CORRUPT_CHECKPOINT_SUFFIX = ".corrupt"
DATASET_SPECS = {
    "mnist": {
        "input_channels": 1,
        "num_classes": 10,
        "normalize_mean": (0.1307,),
        "normalize_std": (0.3081,),
        "dataset_cls": datasets.MNIST,
    },
    "cifar10": {
        "input_channels": 3,
        "num_classes": 10,
        "normalize_mean": (0.4914, 0.4822, 0.4465),
        "normalize_std": (0.2470, 0.2435, 0.2616),
        "dataset_cls": datasets.CIFAR10,
    },
    "cifar100": {
        "input_channels": 3,
        "num_classes": 100,
        "normalize_mean": (0.5071, 0.4867, 0.4408),
        "normalize_std": (0.2675, 0.2565, 0.2761),
        "dataset_cls": datasets.CIFAR100,
    },
}
DATASET_CHOICES = tuple(DATASET_SPECS)


def client_num_threads(num_cpus):
    return max(1, math.ceil(float(num_cpus)))


def configure_thread_env(num_threads):
    for env_var in THREAD_ENV_VARS:
        os.environ[env_var] = str(num_threads)


def configure_torch_threads(num_threads):
    configure_thread_env(num_threads)
    torch.set_num_threads(num_threads)
    return torch.get_num_threads()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


def parse_int_tuple(value, expected_len, option_name):
    if isinstance(value, str):
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    else:
        values = tuple(int(item) for item in value)
    if len(values) != expected_len:
        raise ValueError(f"{option_name} must contain {expected_len} comma-separated integers")
    if any(item < 1 for item in values):
        raise ValueError(f"{option_name} values must all be positive")
    return values


def normalize_resnet_config(
    resnet_depth=18,
    resnet_block="",
    resnet_blocks="",
    resnet_widths=DEFAULT_RESNET_WIDTHS,
    resnet_split_after="layer1",
    resnet_stem_kernel=7,
    resnet_stem_stride=2,
    architecture=None,
    depth=None,
    block=None,
    blocks=None,
    widths=None,
    split_after=None,
    stem_kernel=None,
    stem_stride=None,
):
    if architecture is not None and architecture != "split_resnet":
        raise ValueError(f"Unsupported model architecture: {architecture}")
    if depth is not None:
        resnet_depth = depth
    if block is not None:
        resnet_block = block
    if blocks is not None:
        resnet_blocks = blocks
    if widths is not None:
        resnet_widths = widths
    if split_after is not None:
        resnet_split_after = split_after
    if stem_kernel is not None:
        resnet_stem_kernel = stem_kernel
    if stem_stride is not None:
        resnet_stem_stride = stem_stride

    depth = int(resnet_depth)
    if depth not in RESNET_DEPTH_PRESETS:
        choices = ", ".join(str(item) for item in sorted(RESNET_DEPTH_PRESETS))
        raise ValueError(f"--resnet-depth must be one of: {choices}")

    preset_block, preset_blocks = RESNET_DEPTH_PRESETS[depth]
    block = (resnet_block or preset_block).lower()
    if block not in ("basic", "bottleneck"):
        raise ValueError("--resnet-block must be one of: basic, bottleneck")

    blocks = (
        parse_int_tuple(resnet_blocks, 4, "--resnet-blocks")
        if resnet_blocks
        else tuple(preset_blocks)
    )
    widths = parse_int_tuple(resnet_widths, 4, "--resnet-widths")

    split_after = resnet_split_after.lower()
    if split_after not in RESNET_SPLIT_POINTS:
        raise ValueError(
            "--resnet-split-after must be one of: "
            + ", ".join(RESNET_SPLIT_POINTS)
        )

    stem_kernel = int(resnet_stem_kernel)
    stem_stride = int(resnet_stem_stride)
    if stem_kernel < 1 or stem_kernel % 2 == 0:
        raise ValueError("--resnet-stem-kernel must be a positive odd integer")
    if stem_stride < 1:
        raise ValueError("--resnet-stem-stride must be positive")

    return {
        "architecture": "split_resnet",
        "depth": depth,
        "block": block,
        "blocks": blocks,
        "widths": widths,
        "split_after": split_after,
        "stem_kernel": stem_kernel,
        "stem_stride": stem_stride,
    }


def normalize_cnn_config(cnn_split_after="conv1", architecture=None, split_after=None):
    if architecture is not None and architecture != "split_cnn":
        raise ValueError(f"Unsupported model architecture: {architecture}")
    if split_after is not None:
        cnn_split_after = split_after

    split_after = cnn_split_after.lower()
    if split_after not in CNN_SPLIT_POINTS:
        raise ValueError(
            "--cnn-split-after must be one of: "
            + ", ".join(CNN_SPLIT_POINTS)
        )

    return {
        "architecture": "split_cnn",
        "split_after": split_after,
    }


def normalize_model_config(model_config=None):
    model_config = model_config or {}
    architecture = model_config.get("architecture", "split_resnet")
    if architecture == "split_cnn":
        return normalize_cnn_config(**model_config)
    return normalize_resnet_config(**model_config)


def get_resnet_model_config_from_args(args):
    if getattr(args, "model", "resnet") == "cnn":
        return normalize_cnn_config(cnn_split_after=args.cnn_split_after)
    return normalize_resnet_config(
        resnet_depth=args.resnet_depth,
        resnet_block=args.resnet_block,
        resnet_blocks=args.resnet_blocks,
        resnet_widths=args.resnet_widths,
        resnet_split_after=args.resnet_split_after,
        resnet_stem_kernel=args.resnet_stem_kernel,
        resnet_stem_stride=args.resnet_stem_stride,
    )


def add_resnet_model_args(parser, default_split_after="layer1"):
    parser.add_argument("--model", choices=("resnet", "cnn"), default="resnet")
    parser.add_argument("--resnet-depth", type=int, default=18)
    parser.add_argument(
        "--resnet-block",
        choices=("basic", "bottleneck"),
        default="",
        help="Override the block type selected by --resnet-depth.",
    )
    parser.add_argument(
        "--resnet-blocks",
        default="",
        help="Comma-separated block counts for layer1..layer4, e.g. 2,2,2,2.",
    )
    parser.add_argument(
        "--resnet-widths",
        default="64,128,256,512",
        help="Comma-separated base widths for layer1..layer4.",
    )
    parser.add_argument(
        "--resnet-split-after",
        choices=RESNET_SPLIT_POINTS,
        default=default_split_after,
        help="Split client/server model after this ResNet layer.",
    )
    parser.add_argument("--resnet-stem-kernel", type=int, default=7)
    parser.add_argument("--resnet-stem-stride", type=int, default=2)
    parser.add_argument(
        "--cnn-split-after",
        choices=CNN_SPLIT_POINTS,
        default="conv1",
        help="Split client/server model after this CNN block.",
    )
    return parser


def resnet_block_cls(block_name):
    if block_name == "basic":
        return BasicBlock
    if block_name == "bottleneck":
        return Bottleneck
    raise ValueError(f"Unknown ResNet block: {block_name}")


def make_resnet_layer(block_cls, in_channels, out_channels, num_blocks, stride=1):
    layers = [block_cls(in_channels, out_channels, stride)]
    next_in_channels = out_channels * block_cls.expansion
    for _ in range(1, num_blocks):
        layers.append(block_cls(next_in_channels, out_channels))
    return nn.Sequential(*layers)


def resnet_layer_in_channels(config, layer_index):
    if layer_index == 0:
        return config["widths"][0]
    block_cls = resnet_block_cls(config["block"])
    return config["widths"][layer_index - 1] * block_cls.expansion


def make_configured_resnet_layer(config, layer_index):
    in_channels = resnet_layer_in_channels(config, layer_index)
    out_channels = config["widths"][layer_index]
    stride = 1 if layer_index == 0 else 2
    return make_resnet_layer(
        resnet_block_cls(config["block"]),
        in_channels,
        out_channels,
        config["blocks"][layer_index],
        stride=stride,
    )


class SplitResNetClientNet(nn.Module):
    def __init__(self, input_channels, model_config=None):
        super().__init__()
        self.config = normalize_resnet_config(**(model_config or {}))
        stem_padding = self.config["stem_kernel"] // 2
        self.conv1 = nn.Conv2d(
            input_channels,
            self.config["widths"][0],
            kernel_size=self.config["stem_kernel"],
            stride=self.config["stem_stride"],
            padding=stem_padding,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.config["widths"][0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        split_index = RESNET_SPLIT_POINTS.index(self.config["split_after"])
        for layer_index in range(split_index + 1):
            setattr(
                self,
                f"layer{layer_index + 1}",
                make_configured_resnet_layer(self.config, layer_index),
            )

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        split_index = RESNET_SPLIT_POINTS.index(self.config["split_after"])
        for layer_index in range(split_index + 1):
            x = getattr(self, f"layer{layer_index + 1}")(x)
        return x


class SplitResNetServerNet(nn.Module):
    def __init__(self, model_config=None, num_classes=10):
        super().__init__()
        self.config = normalize_resnet_config(**(model_config or {}))
        split_index = RESNET_SPLIT_POINTS.index(self.config["split_after"])
        for layer_index in range(split_index + 1, len(RESNET_SPLIT_POINTS)):
            setattr(
                self,
                f"layer{layer_index + 1}",
                make_configured_resnet_layer(self.config, layer_index),
            )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        block_cls = resnet_block_cls(self.config["block"])
        self.fc = nn.Linear(self.config["widths"][-1] * block_cls.expansion, num_classes)

    def forward(self, smashed_data):
        x = smashed_data
        split_index = RESNET_SPLIT_POINTS.index(self.config["split_after"])
        for layer_index in range(split_index + 1, len(RESNET_SPLIT_POINTS)):
            x = getattr(self, f"layer{layer_index + 1}")(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class SplitCNNClientNet(nn.Module):
    def __init__(self, input_channels, model_config=None):
        super().__init__()
        self.config = normalize_cnn_config(**(model_config or {}))
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        x = self.conv1(x)
        if self.config["split_after"] == "conv2":
            x = self.conv2(x)
        return x


class SplitCNNServerNet(nn.Module):
    def __init__(self, model_config=None, num_classes=10):
        super().__init__()
        self.config = normalize_cnn_config(**(model_config or {}))
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, smashed_data):
        x = smashed_data
        if self.config["split_after"] == "conv1":
            x = self.conv2(x)
        x = self.pool(x)
        return self.classifier(x)


ClientNet = partial(SplitResNetClientNet, input_channels=1)
CifarClientNet = partial(SplitResNetClientNet, input_channels=3)
ServerNet = SplitResNetServerNet


class FullNet(nn.Module):
    def __init__(self, client_model_cls=ClientNet, server_model_cls=ServerNet):
        super().__init__()
        self.client_model = client_model_cls()
        self.server_model = server_model_cls()

    def forward(self, x):
        return self.server_model(self.client_model(x))


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_mnist_dataset(train):
    return load_dataset("mnist", train)


def normalize_dataset_name(dataset_name):
    normalized = dataset_name.lower().replace("-", "")
    if normalized in DATASET_SPECS:
        return normalized
    raise ValueError("--dataset must be one of: " + ", ".join(DATASET_CHOICES))


def dataset_num_classes(dataset_name):
    dataset_name = normalize_dataset_name(dataset_name)
    return DATASET_SPECS[dataset_name]["num_classes"]


def get_model_classes(dataset_name, model_config=None):
    dataset_name = normalize_dataset_name(dataset_name)
    normalized_config = normalize_model_config(model_config)
    dataset_spec = DATASET_SPECS[dataset_name]
    if normalized_config["architecture"] == "split_cnn":
        return (
            partial(
                SplitCNNClientNet,
                input_channels=dataset_spec["input_channels"],
                model_config=normalized_config,
            ),
            partial(
                SplitCNNServerNet,
                model_config=normalized_config,
                num_classes=dataset_spec["num_classes"],
            ),
        )
    return (
        partial(
            SplitResNetClientNet,
            input_channels=dataset_spec["input_channels"],
            model_config=normalized_config,
        ),
        partial(
            SplitResNetServerNet,
            model_config=normalized_config,
            num_classes=dataset_spec["num_classes"],
        ),
    )


def load_dataset(dataset_name, train):
    dataset_name = normalize_dataset_name(dataset_name)
    cache_key = (dataset_name, bool(train))
    if cache_key in _DATASET_CACHE:
        return _DATASET_CACHE[cache_key]

    dataset_spec = DATASET_SPECS[dataset_name]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            dataset_spec["normalize_mean"],
            dataset_spec["normalize_std"],
        ),
    ])
    dataset = dataset_spec["dataset_cls"](
        root="./data",
        train=train,
        download=True,
        transform=transform,
    )
    _DATASET_CACHE[cache_key] = dataset
    return dataset


def validate_noniid_alpha(noniid_alpha):
    alpha = float(noniid_alpha)
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("--noniid-alpha must be in the range [0, 1]")
    return alpha


def partition_config(num_clients, noniid_alpha, dataset_name="mnist", model_config=None):
    return {
        "dataset": normalize_dataset_name(dataset_name),
        "num_clients": int(num_clients),
        "noniid_alpha": validate_noniid_alpha(noniid_alpha),
        "seed": SEED,
        "model": normalize_model_config(model_config),
    }


def split_indices_iid(dataset_size, num_clients):
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(dataset_size)
    return np.array_split(indices, num_clients)


def split_indices_dirichlet(dataset, num_clients, noniid_alpha):
    alpha = max(validate_noniid_alpha(noniid_alpha), MIN_DIRICHLET_ALPHA)
    rng = np.random.default_rng(SEED)
    targets = np.asarray(dataset.targets)
    class_indices = []

    for class_id in np.unique(targets):
        indices = np.where(targets == class_id)[0]
        rng.shuffle(indices)
        proportions = rng.dirichlet(np.full(num_clients, alpha))
        split_points = (np.cumsum(proportions)[:-1] * len(indices)).astype(int)
        class_indices.extend(np.array_split(indices, split_points))

    client_indices = [[] for _ in range(num_clients)]
    for client_id, indices in enumerate(class_indices):
        client_indices[client_id % num_clients].extend(indices.tolist())

    for client_id in range(num_clients):
        rng.shuffle(client_indices[client_id])

    empty_client_ids = [
        client_id
        for client_id, indices in enumerate(client_indices)
        if not indices
    ]
    for empty_client_id in empty_client_ids:
        donor_client_id = max(
            range(num_clients),
            key=lambda client_id: len(client_indices[client_id]),
        )
        if len(client_indices[donor_client_id]) <= 1:
            break
        client_indices[empty_client_id].append(client_indices[donor_client_id].pop())

    return [np.asarray(indices, dtype=np.int64) for indices in client_indices]


def split_indices(dataset, num_clients, noniid_alpha=1.0):
    alpha = validate_noniid_alpha(noniid_alpha)
    cache_key = (id(dataset), num_clients, alpha)
    if cache_key not in _SPLIT_CACHE:
        if alpha >= 1.0:
            _SPLIT_CACHE[cache_key] = split_indices_iid(len(dataset), num_clients)
        else:
            _SPLIT_CACHE[cache_key] = split_indices_dirichlet(
                dataset,
                num_clients,
                alpha,
            )
    return _SPLIT_CACHE[cache_key]


def client_size(num_clients, client_id, noniid_alpha=1.0, dataset_name="mnist"):
    dataset = load_dataset(dataset_name, train=True)
    return len(split_indices(dataset, num_clients, noniid_alpha)[client_id])


def client_subset(client_id, num_clients, noniid_alpha=1.0, dataset_name="mnist"):
    dataset = load_dataset(dataset_name, train=True)
    indices = split_indices(dataset, num_clients, noniid_alpha)[client_id]
    return Subset(dataset, indices)


def get_batch(
    client_id,
    num_clients,
    batch_index,
    batch_size,
    device,
    noniid_alpha=1.0,
    dataset_name="mnist",
):
    dataset = load_dataset(dataset_name, train=True)
    indices = split_indices(dataset, num_clients, noniid_alpha)[client_id]
    start = batch_index * batch_size
    end = min(start + batch_size, len(indices))
    if start >= len(indices):
        raise IndexError(f"batch_index {batch_index} is outside client {client_id} data")

    x, y = default_collate([dataset[int(index)] for index in indices[start:end]])
    return x.to(device), y.to(device)


def client_label_boundary_score(
    client_id,
    num_clients,
    noniid_alpha=1.0,
    num_classes=None,
    dataset_name="mnist",
):
    dataset = load_dataset(dataset_name, train=True)
    indices = split_indices(dataset, num_clients, noniid_alpha)[client_id]
    if len(indices) == 0:
        return 1.0

    if num_classes is None:
        num_classes = dataset_num_classes(dataset_name)

    targets = np.asarray(dataset.targets)
    labels = targets[np.asarray(indices, dtype=np.int64)]
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    distribution = counts / counts.sum()
    reference = np.full(num_classes, 1.0 / num_classes, dtype=np.float64)
    mixture = 0.5 * (distribution + reference)
    return 0.5 * _kl_divergence_log2(distribution, mixture) + 0.5 * _kl_divergence_log2(reference, mixture)


def _kl_divergence_log2(left, right):
    mask = left > 0
    return float(np.sum(left[mask] * np.log2(left[mask] / right[mask])))


def load_data(
    client_id,
    num_clients,
    train_batch_size=32,
    test_batch_size=128,
    noniid_alpha=1.0,
    dataset_name="mnist",
):
    trainloader = DataLoader(
        client_subset(
            client_id,
            num_clients,
            noniid_alpha=noniid_alpha,
            dataset_name=dataset_name,
        ),
        batch_size=train_batch_size,
        shuffle=True,
    )
    testloader = DataLoader(
        load_dataset(dataset_name, train=False),
        batch_size=test_batch_size,
        shuffle=False,
    )
    return trainloader, testloader


def fedavg_state_dicts(state_dicts, sizes):
    total_size = sum(sizes)
    avg_state = OrderedDict()

    for key in state_dicts[0].keys():
        first_value = state_dicts[0][key]
        if not torch.is_floating_point(first_value):
            avg_state[key] = first_value.clone()
            continue

        avg_state[key] = sum(
            state_dicts[idx][key] * (sizes[idx] / total_size)
            for idx in range(len(state_dicts))
        )
        if not torch.isfinite(avg_state[key]).all():
            raise FloatingPointError(f"FedAvg produced non-finite values for {key}")

    return avg_state


def nonfinite_state_tensors(state_dict, label):
    bad_keys = []
    for key, value in state_dict.items():
        if torch.is_floating_point(value) and not torch.isfinite(value).all():
            bad_keys.append(f"{label}.{key}")
    return bad_keys


def checkpoint_nonfinite_tensors(client_states, server_state):
    bad_keys = []
    for client_id, client_state in enumerate(client_states):
        bad_keys.extend(nonfinite_state_tensors(client_state, f"client{client_id}"))
    bad_keys.extend(nonfinite_state_tensors(server_state, "server"))
    return bad_keys


def quarantine_unreadable_checkpoint(path, error):
    corrupt_path = path.with_name(path.name + CORRUPT_CHECKPOINT_SUFFIX)
    counter = 1
    while corrupt_path.exists():
        corrupt_path = path.with_name(
            f"{path.name}{CORRUPT_CHECKPOINT_SUFFIX}.{counter}"
        )
        counter += 1

    try:
        os.replace(path, corrupt_path)
    except OSError as replace_error:
        print(
            "Ignoring unreadable checkpoint and starting from fresh weights: "
            f"{path} ({error}). Could not move it aside: {replace_error}"
        )
        return

    print(
        "Ignoring unreadable checkpoint and starting from fresh weights: "
        f"{path} ({error}). Moved to: {corrupt_path}"
    )


def load_split_checkpoint(
    checkpoint_path,
    num_clients,
    device,
    noniid_alpha=1.0,
    dataset_name="mnist",
    model_config=None,
):
    if not checkpoint_path:
        return None, None

    path = Path(checkpoint_path)
    if not path.exists():
        return None, None

    try:
        checkpoint = torch.load(path, map_location=device)
    except Exception as error:
        quarantine_unreadable_checkpoint(path, error)
        return None, None

    client_states = checkpoint["client_models"]
    if len(client_states) != num_clients:
        raise ValueError(
            f"Checkpoint has {len(client_states)} clients, "
            f"but current run has {num_clients} clients"
        )

    expected_partition = partition_config(
        num_clients,
        noniid_alpha,
        dataset_name,
        model_config=model_config,
    )
    saved_partition = checkpoint.get("partition")
    if saved_partition is None:
        print(
            "Loaded legacy checkpoint without partition metadata; "
            "cannot verify non-IID alpha consistency."
        )
    elif "dataset" not in saved_partition:
        if normalize_dataset_name(dataset_name) != "mnist":
            raise ValueError(
                "Checkpoint has no dataset metadata and can only be reused with MNIST"
            )
        legacy_expected = {
            "num_clients": expected_partition["num_clients"],
            "noniid_alpha": expected_partition["noniid_alpha"],
            "seed": expected_partition["seed"],
        }
        if saved_partition != legacy_expected:
            raise ValueError(
                "Checkpoint partition config does not match current run: "
                f"checkpoint={saved_partition}, current={legacy_expected}"
            )
        print("Loaded legacy MNIST checkpoint without dataset metadata.")
    elif "model" not in saved_partition:
        legacy_expected = {
            key: expected_partition[key]
            for key in ("dataset", "num_clients", "noniid_alpha", "seed")
        }
        saved_without_model = {
            key: saved_partition[key]
            for key in ("dataset", "num_clients", "noniid_alpha", "seed")
        }
        if saved_without_model != legacy_expected:
            raise ValueError(
                "Checkpoint partition config does not match current run: "
                f"checkpoint={saved_without_model}, current={legacy_expected}"
            )
        print(
            "Loaded checkpoint without model metadata; "
            "cannot verify ResNet config consistency."
        )
    elif saved_partition != expected_partition:
        raise ValueError(
            "Checkpoint partition config does not match current run: "
            f"checkpoint={saved_partition}, current={expected_partition}"
        )

    bad_keys = checkpoint_nonfinite_tensors(client_states, checkpoint["server_model"])
    if bad_keys:
        quarantine_unreadable_checkpoint(
            path,
            "checkpoint contains non-finite tensor values: "
            + ", ".join(bad_keys[:10])
            + (" ..." if len(bad_keys) > 10 else ""),
        )
        return None, None

    print(f"Loaded checkpoint: {path}")
    return client_states, checkpoint["server_model"]


def save_split_checkpoint(
    checkpoint_path,
    client_states,
    server_model,
    num_clients=None,
    noniid_alpha=1.0,
    dataset_name="mnist",
    model_config=None,
):
    if not checkpoint_path:
        return

    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(
            {
                "client_models": [copy.deepcopy(state) for state in client_states],
                "server_model": copy.deepcopy(server_model.state_dict()),
                "partition": (
                    partition_config(
                        num_clients,
                        noniid_alpha,
                        dataset_name,
                        model_config=model_config,
                    )
                    if num_clients is not None
                    else None
                ),
            },
            temp_path,
        )
        os.replace(temp_path, path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise
    print(f"Saved checkpoint: {path}")


def check_simulation_backend(
    num_clients,
    client_num_cpus,
    simulation_name="Flower simulation",
):
    print("Checking Flower simulation backend...", flush=True)
    requested_cpus = max(1, math.ceil(num_clients * client_num_cpus))
    total_num_cpus = simulation_total_num_cpus()
    try:
        import ray  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"{simulation_name} requires the Ray backend, but `ray` is not "
            "installed for this Python environment. On Windows, Ray is not "
            "available for all Python versions; use a Python version with a Ray "
            "wheel, or run this in WSL2."
        ) from exc

    print(f"Ray total CPU slots: {total_num_cpus}", flush=True)
    print(f"Requested client CPU slots: {requested_cpus}", flush=True)
    if requested_cpus > total_num_cpus:
        print(
            f"Requested client CPU slots exceed Ray total CPU slots; "
            f"clients will run in waves.",
            flush=True,
        )

    return total_num_cpus


def simulation_total_num_cpus():
    configured_total_cpus = os.environ.get(SIMULATION_TOTAL_CPUS_ENV)
    if configured_total_cpus:
        total_num_cpus = int(configured_total_cpus)
        if total_num_cpus < 1:
            raise ValueError(f"{SIMULATION_TOTAL_CPUS_ENV} must be at least 1")
        return total_num_cpus

    return max(1, torch.get_num_threads())


def build_ray_backend_config(total_num_cpus, client_num_cpus, client_num_gpus=0.0):
    return {
        "init_args": {
            "num_cpus": total_num_cpus,
            "include_dashboard": False,
        },
        "client_resources": {
            "num_cpus": client_num_cpus,
            "num_gpus": client_num_gpus,
        },
    }
