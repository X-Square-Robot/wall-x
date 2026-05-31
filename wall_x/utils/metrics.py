import torch
from typing import List


def get_action_accuracy(
    gt: torch.FloatTensor,  # [Batch_Size, Horizon, Action_Dim]
    pred: torch.FloatTensor,
    thresholds: List[float] = [0.1, 0.2],
) -> torch.FloatTensor:
    device = gt.device
    diff = torch.abs(gt - pred).reshape(-1, gt.shape[-1])

    # get the percentage of diff lower than threshold for all action dimensions
    accuracies = torch.zeros(len(thresholds), device=device)
    for idx, threshold in enumerate(thresholds):
        accuracy = torch.mean(
            (torch.mean((diff < threshold).float(), dim=1) >= 1.0).float()
        )
        accuracies[idx] = accuracy
    return accuracies


def dtw_distance(seq1, seq2):
    """
    Compute the Dynamic Time Warping distance between two sequences.

    ``seq1`` and ``seq2`` must have shape ``(T, D)``, where ``T`` is the
    number of time steps and ``D`` is the feature dimension.
    """
    n, m = seq1.shape[0], seq2.shape[0]

    seq1_d, seq2_d = seq1.double(), seq2.double()
    dtw_matrix = torch.full(
        (n + 1, m + 1), float("inf"), dtype=torch.float64, device=seq1.device
    )
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = torch.dist(seq1_d[i - 1], seq2_d[j - 1])
            dtw_matrix[i, j] = cost + torch.min(
                torch.stack(
                    [
                        dtw_matrix[i - 1, j],
                        dtw_matrix[i, j - 1],
                        dtw_matrix[i - 1, j - 1],
                    ]
                )
            )

    return dtw_matrix[n, m]


def frechet_distance(seq1, seq2):
    """
    Compute the discrete Frechet distance between two sequences.

    ``seq1`` and ``seq2`` must have shape ``(T, D)``, where ``T`` is the
    number of time steps and ``D`` is the feature dimension.
    """
    n, m = seq1.shape[0], seq2.shape[0]

    seq1_d, seq2_d = seq1.double(), seq2.double()
    frechet_matrix = torch.full(
        (n, m), float("inf"), dtype=torch.float64, device=seq1.device
    )
    frechet_matrix[0, 0] = torch.dist(seq1_d[0], seq2_d[0])

    for j in range(1, m):
        frechet_matrix[0, j] = torch.max(
            frechet_matrix[0, j - 1], torch.dist(seq1_d[0], seq2_d[j])
        )

    for i in range(1, n):
        frechet_matrix[i, 0] = torch.max(
            frechet_matrix[i - 1, 0], torch.dist(seq1_d[i], seq2_d[0])
        )

    for i in range(1, n):
        for j in range(1, m):
            frechet_matrix[i, j] = torch.max(
                torch.min(
                    torch.stack(
                        [
                            frechet_matrix[i - 1, j],
                            frechet_matrix[i, j - 1],
                            frechet_matrix[i - 1, j - 1],
                        ]
                    )
                ),
                torch.dist(seq1_d[i], seq2_d[j]),
            )

    return frechet_matrix[n - 1, m - 1]
