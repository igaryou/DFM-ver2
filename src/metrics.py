from __future__ import annotations

import torch


class SegmentationMetrics:
    """20-way confusion matrix; GT void is filtered, predicted void is retained."""

    def __init__(
        self,
        num_classes: int = 20,
        void_class_index: int = 19,
        device: torch.device | str = "cpu",
        evaluated_class_indices: list[int] | range | None = None,
        nanmean: bool = False,
    ) -> None:
        self.num_classes = num_classes
        self.void_class_index = void_class_index
        self.evaluated_class_indices = (
            list(evaluated_class_indices)
            if evaluated_class_indices is not None
            else [index for index in range(num_classes) if index != void_class_index]
        )
        self.nanmean = nanmean
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
            "prediction_void_retained": True,
        }
