from __future__ import annotations

import math

import numpy as np
import torch
import torch.distributed as dist


def binary_pairwise_iou(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Pairwise IoU for [B,N,H,W] and [B,G,H,W] binary masks.

    Two empty masks have IoU 1: they agree perfectly and never produce NaN.
    """
    prediction = prediction.bool()
    target = target.bool()
    if prediction.ndim != 4 or target.ndim != 4:
        raise ValueError("binary mask sets must have shape [B,S,H,W]")
    if prediction.shape[0] != target.shape[0] or prediction.shape[-2:] != target.shape[-2:]:
        raise ValueError("prediction and target mask sets must share batch/spatial shape")
    prediction_flat = prediction.flatten(2)
    target_flat = target.flatten(2)
    intersection = torch.bmm(
        prediction_flat.float(), target_flat.float().transpose(1, 2)
    ).double()
    union = (
        prediction_flat.sum(dim=-1)[:, :, None]
        + target_flat.sum(dim=-1)[:, None, :]
        - intersection
    ).double()
    return torch.where(union > 0, intersection / union, torch.ones_like(union))


def binary_pairwise_dice(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Pairwise Dice with Dice(empty, empty)=1."""
    prediction = prediction.bool()
    target = target.bool()
    if prediction.ndim != 4 or target.ndim != 4:
        raise ValueError("binary mask sets must have shape [B,S,H,W]")
    if prediction.shape[0] != target.shape[0] or prediction.shape[-2:] != target.shape[-2:]:
        raise ValueError("prediction and target mask sets must share batch/spatial shape")
    prediction_flat = prediction.flatten(2)
    target_flat = target.flatten(2)
    intersection = torch.bmm(
        prediction_flat.float(), target_flat.float().transpose(1, 2)
    ).double()
    size = (
        prediction_flat.sum(dim=-1)[:, :, None]
        + target_flat.sum(dim=-1)[:, None, :]
    ).double()
    return torch.where(size > 0, 2.0 * intersection / size, torch.ones_like(size))


def _maximum_one_to_one_mean(score: torch.Tensor) -> torch.Tensor:
    """Exact max-weight matching for [B,N,G], optimized for LIDC's G=4."""
    if score.ndim != 3:
        raise ValueError("matching score must have shape [B,N,G]")
    batch, predictions, ground_truths = score.shape
    if predictions < ground_truths:
        score = score.transpose(1, 2)
        batch, predictions, ground_truths = score.shape
    states = 1 << ground_truths
    dp = score.new_full((batch, states), -torch.inf)
    dp[:, 0] = 0.0
    source_indices = []
    destination_indices = []
    ground_truth_indices = []
    for mask in range(states):
        for ground_truth_index in range(ground_truths):
            bit = 1 << ground_truth_index
            if not mask & bit:
                source_indices.append(mask)
                destination_indices.append(mask | bit)
                ground_truth_indices.append(ground_truth_index)
    source_indices = torch.tensor(source_indices, device=score.device)
    destination_indices = torch.tensor(destination_indices, device=score.device)
    ground_truth_indices = torch.tensor(ground_truth_indices, device=score.device)
    for prediction_index in range(predictions):
        previous = dp
        updated = previous.clone()
        candidate = previous[:, source_indices] + score[
            :, prediction_index, ground_truth_indices
        ]
        updated.scatter_reduce_(
            1,
            destination_indices[None].expand(batch, -1),
            candidate,
            reduce="amax",
            include_self=True,
        )
        dp = updated
    return dp[:, -1] / ground_truths


def _hungarian_maximum_assignment_mean(score: torch.Tensor) -> torch.Tensor:
    """Solve an exact maximum-weight assignment for each square score matrix."""
    if score.ndim != 3 or score.shape[1] != score.shape[2]:
        raise ValueError("Hungarian score must have shape [B,L,L]")
    size = score.shape[1]
    results = []
    # Evaluation metric only: the O(L^3) Hungarian updates are intentionally
    # performed on detached CPU doubles for clarity and deterministic indexing.
    for batch_score_tensor in score.detach().double().cpu():
        batch_score = batch_score_tensor.numpy()
        cost = -batch_score
        row_potential = np.zeros(size + 1, dtype=np.float64)
        column_potential = np.zeros(size + 1, dtype=np.float64)
        matched_row = np.zeros(size + 1, dtype=np.int64)
        predecessor = np.zeros(size + 1, dtype=np.int64)
        for row in range(1, size + 1):
            matched_row[0] = row
            minimum = np.full(size + 1, np.inf, dtype=np.float64)
            used = np.zeros(size + 1, dtype=np.bool_)
            column = 0
            while True:
                used[column] = True
                current_row = matched_row[column]
                delta = float("inf")
                next_column = 0
                for candidate in range(1, size + 1):
                    if used[candidate]:
                        continue
                    reduced = (
                        cost[current_row - 1, candidate - 1]
                        - row_potential[current_row]
                        - column_potential[candidate]
                    )
                    if reduced < minimum[candidate]:
                        minimum[candidate] = reduced
                        predecessor[candidate] = column
                    candidate_minimum = minimum[candidate]
                    if candidate_minimum < delta:
                        # Keep a Python scalar: a tensor view would be mutated
                        # below when minimum is updated, corrupting potentials.
                        delta = candidate_minimum
                        next_column = candidate
                for candidate in range(size + 1):
                    if used[candidate]:
                        row_potential[matched_row[candidate]] += delta
                        column_potential[candidate] -= delta
                    else:
                        minimum[candidate] -= delta
                column = next_column
                if matched_row[column] == 0:
                    break
            while True:
                previous_column = predecessor[column]
                matched_row[column] = matched_row[previous_column]
                column = previous_column
                if column == 0:
                    break
        assignment = np.empty(size, dtype=np.int64)
        for column in range(1, size + 1):
            assignment[matched_row[column] - 1] = column - 1
        results.append(batch_score[np.arange(size), assignment].mean())
    return score.new_tensor(results)


def ccdm_hungarian_matched_iou(
    predictions: torch.Tensor, ground_truths: torch.Tensor
) -> torch.Tensor:
    """Compute CCDM-compatible HM-IoU for arbitrary prediction/GT counts.

    Predictions and GTs are repeat-interleaved to L=lcm(N,G), then all L
    samples participate in one exact Hungarian one-to-one matching. This
    differs from the legacy best-subset DP, which selected only G predictions.
    """
    if predictions.ndim != 4 or ground_truths.ndim != 4:
        raise ValueError("binary mask sets must have shape [B,S,H,W]")
    predictions_count = predictions.shape[1]
    ground_truth_count = ground_truths.shape[1]
    if predictions_count == 0 or ground_truth_count == 0:
        raise ValueError("prediction and ground-truth sets must not be empty")
    pairwise = binary_pairwise_iou(predictions, ground_truths)
    expanded_count = math.lcm(predictions_count, ground_truth_count)
    expanded = pairwise.repeat_interleave(
        expanded_count // predictions_count, dim=1
    ).repeat_interleave(
        expanded_count // ground_truth_count, dim=2
    )
    assert expanded.shape[1:] == (expanded_count, expanded_count)
    return _hungarian_maximum_assignment_mean(expanded)


class LIDCStochasticMetrics:
    """Batch-streaming LIDC distribution and deterministic binary metrics."""

    def __init__(self, sample_counts: list[int], device: torch.device | str = "cpu"):
        if not sample_counts or sample_counts != sorted(set(sample_counts)):
            raise ValueError("sample_counts must be a non-empty sorted unique list")
        self.sample_counts = sample_counts
        self.device = torch.device(device)
        self.distribution_sums = torch.zeros(
            len(sample_counts), 3, dtype=torch.float64, device=self.device
        )
        self.image_count = torch.zeros((), dtype=torch.float64, device=self.device)
        # TP, FP, FN, TN for the probability-mean prediction against all 4 GTs.
        self.confusion = torch.zeros(4, dtype=torch.float64, device=self.device)

    @torch.no_grad()
    def update(
        self,
        predictions: torch.Tensor,
        ground_truths: torch.Tensor,
        deterministic_prediction: torch.Tensor,
    ) -> None:
        predictions = predictions.to(self.device).bool()
        ground_truths = ground_truths.to(self.device).bool()
        deterministic_prediction = deterministic_prediction.to(self.device).bool()
        if predictions.shape[1] < self.sample_counts[-1]:
            raise ValueError("not enough predictions for configured sample counts")
        if ground_truths.shape[1] != 4:
            raise ValueError("LIDC stochastic evaluation requires exactly 4 GT masks")
        for index, count in enumerate(self.sample_counts):
            selected = predictions[:, :count]
            pred_gt_iou = binary_pairwise_iou(selected, ground_truths)
            pred_pred_iou = binary_pairwise_iou(selected, selected)
            gt_gt_iou = binary_pairwise_iou(ground_truths, ground_truths)
            ged_squared = (
                2.0 * (1.0 - pred_gt_iou).mean(dim=(1, 2))
                - (1.0 - pred_pred_iou).mean(dim=(1, 2))
                - (1.0 - gt_gt_iou).mean(dim=(1, 2))
            )
            # CCDM HM-IoU includes every sample after LCM expansion, rather
            # than selecting only the best G predictions from N.
            hm_iou = ccdm_hungarian_matched_iou(selected, ground_truths)
            mdm = binary_pairwise_dice(selected, ground_truths).amax(dim=1).mean(dim=1)
            self.distribution_sums[index] += torch.stack(
                (ged_squared.sum(), hm_iou.sum(), mdm.sum())
            )
        expanded_prediction = deterministic_prediction[:, None].expand_as(ground_truths)
        tp = (expanded_prediction & ground_truths).sum()
        fp = (expanded_prediction & ~ground_truths).sum()
        fn = (~expanded_prediction & ground_truths).sum()
        tn = (~expanded_prediction & ~ground_truths).sum()
        self.confusion += torch.stack((tp, fp, fn, tn)).double()
        self.image_count += predictions.shape[0]

    @torch.no_grad()
    def synchronize(self) -> None:
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(self.distribution_sums)
            dist.all_reduce(self.image_count)
            dist.all_reduce(self.confusion)

    def compute(self) -> dict[str, float]:
        count = self.image_count.clamp_min(1.0)
        result: dict[str, float] = {}
        for index, samples in enumerate(self.sample_counts):
            result[f"ged{samples}"] = float(self.distribution_sums[index, 0] / count)
            result[f"hm_iou{samples}"] = float(self.distribution_sums[index, 1] / count)
            result[f"mdm{samples}"] = float(self.distribution_sums[index, 2] / count)
        tp, fp, fn, tn = self.confusion

        def ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
            # Empty-vs-empty is a perfect binary result, consistent with IoU above.
            return float(torch.where(denominator > 0, numerator / denominator, 1.0))

        result.update({
            "dice": ratio(2 * tp, 2 * tp + fp + fn),
            "foreground_iou": ratio(tp, tp + fp + fn),
            "precision": ratio(tp, tp + fp),
            "sensitivity": ratio(tp, tp + fn),
            "specificity": ratio(tn, tn + fp),
        })
        return result


class SegmentationMetrics:
    """Full-state confusion matrix with GT void filtered from evaluation."""

    def __init__(
        self,
        num_classes: int = 20,
        void_class_index: int = 19,
        device: torch.device | str = "cpu",
        evaluated_class_indices: list[int] | range | None = None,
        nanmean: bool = False,
        prediction_void_retained: bool = True,
    ) -> None:
        self.num_classes = num_classes
        self.void_class_index = void_class_index
        self.evaluated_class_indices = (
            list(evaluated_class_indices)
            if evaluated_class_indices is not None
            else [index for index in range(num_classes) if index != void_class_index]
        )
        self.nanmean = nanmean
        self.prediction_void_retained = prediction_void_retained
        self.confusion_matrix = torch.zeros(
            num_classes, num_classes, dtype=torch.int64, device=device
        )

    @torch.no_grad()
    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        device = self.confusion_matrix.device
        prediction = prediction.detach().reshape(-1).to(device)
        target = target.detach().reshape(-1).to(device)
        valid = (
            (target >= 0) & (target < self.num_classes)
            & (target != self.void_class_index)
            & (prediction >= 0) & (prediction < self.num_classes)
        )
        indices = target[valid] * self.num_classes + prediction[valid]
        self.confusion_matrix += torch.bincount(
            indices, minlength=self.num_classes**2
        ).reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict:
        confusion = self.confusion_matrix.float()
        true_positive = confusion.diag()
        ground_truth = confusion.sum(dim=1)
        predicted = confusion.sum(dim=0)
        union = ground_truth + predicted - true_positive
        if self.nanmean:
            iou = torch.where(union > 0, true_positive / union, torch.nan)
            class_accuracy = torch.where(
                ground_truth > 0, true_positive / ground_truth, torch.nan
            )
        else:
            iou = true_positive / union.clamp_min(1.0)
            class_accuracy = true_positive / ground_truth.clamp_min(1.0)
        evaluated = torch.tensor(
            self.evaluated_class_indices, device=confusion.device, dtype=torch.long
        )
        mean = torch.nanmean if self.nanmean else torch.mean
        return {
            "mIoU": float(mean(iou[evaluated])),
            "pixel_acc": float(true_positive.sum() / confusion.sum().clamp_min(1.0)),
            "mAcc": float(mean(class_accuracy[evaluated])),
            "class_iou": [float(value) for value in iou[evaluated]],
            "class_accuracy": [float(value) for value in class_accuracy[evaluated]],
            "confusion_matrix": self.confusion_matrix.cpu().tolist(),
            "evaluated_class_indices": self.evaluated_class_indices,
            "void_gt_excluded": self.void_class_index,
            "prediction_void_retained": self.prediction_void_retained,
        }
