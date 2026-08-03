import argparse
from dataclasses import dataclass

import numpy as np

from split_learning_utils import load_mnist_dataset, split_indices, validate_noniid_alpha


# JSD is computed with log2, so pairwise values are in [0, 1].
DEFAULT_IID_JSD_THRESHOLD = 0.05
DEFAULT_STRONG_NONIID_JSD_THRESHOLD = 0.20


@dataclass(frozen=True)
class NoniidCondition:
    label_counts: np.ndarray
    label_distributions: np.ndarray
    jsd_matrix: np.ndarray
    mean_jsd: float
    max_jsd: float
    condition: str
    empty_client_ids: tuple[int, ...]


def client_label_counts(dataset, num_clients, noniid_alpha=1.0, num_classes=None):
    if num_clients < 1:
        raise ValueError("num_clients must be at least 1")

    targets = np.asarray(dataset.targets)
    if num_classes is None:
        num_classes = int(targets.max()) + 1

    counts = np.zeros((num_clients, num_classes), dtype=np.float64)
    client_indices = split_indices(dataset, num_clients, noniid_alpha)
    for client_id, indices in enumerate(client_indices):
        labels = targets[np.asarray(indices, dtype=np.int64)]
        counts[client_id] = np.bincount(labels, minlength=num_classes)
    return counts


def label_distributions(label_counts):
    counts = np.asarray(label_counts, dtype=np.float64)
    totals = counts.sum(axis=1, keepdims=True)
    distributions = np.full_like(counts, np.nan)
    return np.divide(
        counts,
        totals,
        out=distributions,
        where=totals > 0,
    )


def jensen_shannon_divergence(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if not _is_probability_distribution(left) or not _is_probability_distribution(right):
        return np.nan

    mixture = 0.5 * (left + right)
    return 0.5 * _kl_divergence(left, mixture) + 0.5 * _kl_divergence(right, mixture)


def jsd_matrix(distributions):
    distributions = np.asarray(distributions, dtype=np.float64)
    num_clients = distributions.shape[0]
    matrix = np.zeros((num_clients, num_clients), dtype=np.float64)

    for left_id in range(num_clients):
        for right_id in range(left_id + 1, num_clients):
            value = jensen_shannon_divergence(
                distributions[left_id],
                distributions[right_id],
            )
            matrix[left_id, right_id] = value
            matrix[right_id, left_id] = value
    return matrix


def summarize_jsd_matrix(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape[0] < 2:
        return 0.0, 0.0

    upper_triangle = matrix[np.triu_indices(matrix.shape[0], k=1)]
    upper_triangle = upper_triangle[np.isfinite(upper_triangle)]
    if upper_triangle.size == 0:
        return 0.0, 0.0

    return float(upper_triangle.mean()), float(upper_triangle.max())


def classify_noniid_condition(
    mean_jsd,
    empty_client_ids=(),
    iid_threshold=DEFAULT_IID_JSD_THRESHOLD,
    strong_threshold=DEFAULT_STRONG_NONIID_JSD_THRESHOLD,
):
    validate_jsd_thresholds(iid_threshold, strong_threshold)
    if empty_client_ids:
        return "strong_noniid"
    if mean_jsd <= iid_threshold:
        return "iid"
    if mean_jsd >= strong_threshold:
        return "strong_noniid"
    return "mild_noniid"


def detect_noniid_condition(
    num_clients,
    noniid_alpha=1.0,
    dataset=None,
    iid_threshold=DEFAULT_IID_JSD_THRESHOLD,
    strong_threshold=DEFAULT_STRONG_NONIID_JSD_THRESHOLD,
):
    validate_noniid_alpha(noniid_alpha)
    validate_jsd_thresholds(iid_threshold, strong_threshold)
    if dataset is None:
        dataset = load_mnist_dataset(train=True)

    counts = client_label_counts(dataset, num_clients, noniid_alpha)
    empty_client_ids = tuple(
        int(client_id)
        for client_id, count_row in enumerate(counts)
        if count_row.sum() == 0
    )
    distributions = label_distributions(counts)
    matrix = jsd_matrix(distributions)
    mean_jsd, max_jsd = summarize_jsd_matrix(matrix)
    condition = classify_noniid_condition(
        mean_jsd,
        empty_client_ids=empty_client_ids,
        iid_threshold=iid_threshold,
        strong_threshold=strong_threshold,
    )

    return NoniidCondition(
        label_counts=counts,
        label_distributions=distributions,
        jsd_matrix=matrix,
        mean_jsd=mean_jsd,
        max_jsd=max_jsd,
        condition=condition,
        empty_client_ids=empty_client_ids,
    )


def switch_method_for_noniid(
    current_method,
    condition,
    strong_noniid_method="sfl",
):
    empty_client_text = ""
    if condition.empty_client_ids:
        empty_client_text = f", empty_clients={list(condition.empty_client_ids)}"

    if condition.condition == "strong_noniid" and current_method != strong_noniid_method:
        return strong_noniid_method, (
            f"strong non-IID detected by JSD matrix "
            f"(mean={condition.mean_jsd:.4f}, max={condition.max_jsd:.4f}"
            f"{empty_client_text})"
        )
    return current_method, (
        f"non-IID condition is {condition.condition} by JSD matrix "
        f"(mean={condition.mean_jsd:.4f}, max={condition.max_jsd:.4f}"
        f"{empty_client_text})"
    )


def print_noniid_report(condition):
    print("Client label counts:")
    print(condition.label_counts.astype(int))
    print("Client label distributions:")
    print(np.round(condition.label_distributions, 4))
    print("JSD matrix:")
    print(np.round(condition.jsd_matrix, 4))
    print(f"Mean pairwise JSD: {condition.mean_jsd:.4f}")
    print(f"Max pairwise JSD:  {condition.max_jsd:.4f}")
    print(f"Empty clients:     {list(condition.empty_client_ids)}")
    print(f"Condition:         {condition.condition}")


def _kl_divergence(left, right):
    mask = left > 0
    return float(np.sum(left[mask] * np.log2(left[mask] / right[mask])))


def _is_probability_distribution(values):
    return (
        np.all(np.isfinite(values))
        and np.all(values >= 0)
        and np.isclose(values.sum(), 1.0)
    )


def validate_jsd_thresholds(iid_threshold, strong_threshold):
    iid_threshold = float(iid_threshold)
    strong_threshold = float(strong_threshold)
    if iid_threshold < 0.0 or iid_threshold > 1.0:
        raise ValueError("--iid-jsd-threshold must be in the range [0, 1]")
    if strong_threshold < 0.0 or strong_threshold > 1.0:
        raise ValueError("--strong-noniid-jsd-threshold must be in the range [0, 1]")
    if iid_threshold >= strong_threshold:
        raise ValueError(
            "--iid-jsd-threshold must be smaller than "
            "--strong-noniid-jsd-threshold"
        )
    return iid_threshold, strong_threshold


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-clients", type=int, required=True)
    parser.add_argument("--noniid-alpha", type=float, default=1.0)
    parser.add_argument(
        "--iid-jsd-threshold",
        type=float,
        default=DEFAULT_IID_JSD_THRESHOLD,
        help="Mean pairwise log2-JSD at or below this value is treated as IID.",
    )
    parser.add_argument(
        "--strong-noniid-jsd-threshold",
        type=float,
        default=DEFAULT_STRONG_NONIID_JSD_THRESHOLD,
        help="Mean pairwise log2-JSD at or above this value is treated as strong non-IID.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    condition = detect_noniid_condition(
        num_clients=args.num_clients,
        noniid_alpha=args.noniid_alpha,
        iid_threshold=args.iid_jsd_threshold,
        strong_threshold=args.strong_noniid_jsd_threshold,
    )
    print_noniid_report(condition)


if __name__ == "__main__":
    main()
