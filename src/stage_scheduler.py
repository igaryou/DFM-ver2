from __future__ import annotations

import math
from typing import Any


STAGE_AWARE_SCHEDULER_VERSION = 1


def _active_epoch_total(config: dict, target: str) -> int:
    flag = "train_source" if target == "source" else "train_flow"
    stages = config["training"]["stages"]
    return sum(
        int(stage["end_epoch"]) - int(stage["start_epoch"])
        for name in ("source_pretrain", "flow_training")
        for stage in (stages[name],)
        if stage["enabled"] and stage[flag]
    )


class StageAwareGroupLRScheduler:
    """Epoch scheduler with independent active-epoch clocks per logical group."""

    stage_aware = True

    def __init__(self, optimizer, config: dict) -> None:
        self.optimizer = optimizer
        self.config = config
        scheduler = config["training"]["scheduler"]
        self.name = scheduler["name"]
        self.warmup_epochs = int(scheduler["warmup_epochs"])
        self.warmup_start_factor = float(scheduler["warmup_start_factor"])
        self.eta_min = float(scheduler["eta_min"])
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.progress = {"flow": 0, "source": 0}
        self.total_epochs = {
            target: _active_epoch_total(config, target)
            for target in ("flow", "source")
        }
        self.last_epoch = 0
        self._apply_lrs()

    @staticmethod
    def target_for_group(group: dict) -> str:
        return "source" if group.get("name") == "source" else "flow"

    def _target_reference_lr(self, target: str) -> float:
        for base_lr, group in zip(
            self.base_lrs, self.optimizer.param_groups, strict=True
        ):
            if self.target_for_group(group) == target:
                return base_lr
        return self.base_lrs[0]

    def _factor(self, target: str, progress: int) -> float:
        if self.name == "constant":
            return 1.0
        total = max(self.total_epochs[target], 1)
        warmup = min(self.warmup_epochs, total)
        if warmup > 0 and progress < warmup:
            return self.warmup_start_factor + (
                1.0 - self.warmup_start_factor
            ) * progress / warmup
        cosine_total = max(total - warmup, 1)
        cosine_progress = min(max(progress - warmup, 0), cosine_total)
        base_lr = self._target_reference_lr(target)
        eta_ratio = self.eta_min / base_lr if base_lr > 0 else 0.0
        return eta_ratio + (1.0 - eta_ratio) * (
            1.0 + math.cos(math.pi * cosine_progress / cosine_total)
        ) / 2.0

    def _apply_lrs(self) -> None:
        for base_lr, group in zip(
            self.base_lrs, self.optimizer.param_groups, strict=True
        ):
            target = self.target_for_group(group)
            group["lr"] = base_lr * self._factor(
                target, self.progress[target]
            )

    def step(
        self, *, train_flow: bool = True, train_source: bool = True
    ) -> None:
        if train_flow:
            self.progress["flow"] += 1
        if train_source:
            self.progress["source"] += 1
        for target in self.progress:
            self.progress[target] = min(
                self.progress[target], self.total_epochs[target]
            )
        self.last_epoch = max(self.progress.values())
        self._apply_lrs()

    def get_last_lr(self) -> list[float]:
        return [
            float(group["lr"]) for group in self.optimizer.param_groups
        ]

    def progress_for(self, target: str) -> int:
        if target not in self.progress:
            raise ValueError("scheduler target must be flow or source")
        return int(self.progress[target])

    def set_progress_from_completed_epochs(
        self, completed_epochs: int
    ) -> None:
        completed = min(
            max(int(completed_epochs), 0),
            self.config["training"]["epochs"],
        )
        progress = {"flow": 0, "source": 0}
        stages = self.config["training"]["stages"]
        for epoch in range(completed):
            for name in ("source_pretrain", "flow_training"):
                stage = stages[name]
                if not stage["enabled"]:
                    continue
                if (
                    int(stage["start_epoch"])
                    <= epoch
                    < int(stage["end_epoch"])
                ):
                    progress["flow"] += int(stage["train_flow"])
                    progress["source"] += int(stage["train_source"])
                    break
        self.progress = progress
        self.last_epoch = max(progress.values())
        self._apply_lrs()

    def state_dict(self) -> dict[str, Any]:
        return {
            "stage_aware_version": STAGE_AWARE_SCHEDULER_VERSION,
            "progress": dict(self.progress),
            "total_epochs": dict(self.total_epochs),
            "base_lrs": list(self.base_lrs),
            "last_epoch": self.last_epoch,
            "name": self.name,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if "stage_aware_version" not in state_dict:
            raise ValueError(
                "checkpoint contains a legacy global scheduler state"
            )
        saved_totals = {
            key: int(value)
            for key, value in state_dict["total_epochs"].items()
        }
        if saved_totals != self.total_epochs:
            raise RuntimeError(
                "stage-aware scheduler active epoch totals changed: "
                f"checkpoint={saved_totals}, config={self.total_epochs}"
            )
        saved_base_lrs = [
            float(value) for value in state_dict["base_lrs"]
        ]
        if len(saved_base_lrs) != len(self.base_lrs):
            raise RuntimeError(
                "stage-aware scheduler parameter group count changed"
            )
        self.base_lrs = saved_base_lrs
        self.progress = {
            "flow": int(state_dict["progress"]["flow"]),
            "source": int(state_dict["progress"]["source"]),
        }
        self.last_epoch = int(
            state_dict.get("last_epoch", max(self.progress.values()))
        )
        self._apply_lrs()
