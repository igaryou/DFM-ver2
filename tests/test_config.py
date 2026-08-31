from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from checkpoint import model_signature
from config import load_config, save_resolved_config
from trainer import log_esd_experiment_metadata


CONFIG = Path(__file__).parents[1] / "configs" / "debug_diagonal_cityscapes.yaml"
ESD_CONFIG = Path(__file__).parents[1] / "configs" / "debug_esd_cityscapes.yaml"
PSD_FROM_JOINT_CONFIG = (
    Path(__file__).parents[1]
    / "configs"
    / "stage2_psd_from_joint500_cityscapes.yaml"
)
JOINT_PSD_CITYSCAPES_CONFIG = (
    Path(__file__).parents[1] / "configs" / "joint_psd_cityscapes.yaml"
)
DIAGONAL_ADE20K_CONFIG = Path(__file__).parents[1] / "configs" / "diagonal_ade20k.yaml"
EXPECTED_ESD_METADATA = {
    "formulation": "stabilized_logit_space",
    "source": "discrete_flow_maps",
    "additional_numerical_safeguards": True,
}
FULL_TRAINING_CONFIGS = tuple(
    Path(__file__).parents[1] / "configs" / f"{stage}_{loss}_cityscapes.yaml"
    for stage in ("stage2", "joint")
    for loss in ("psd", "csd", "ecld", "esd")
)
FULL_TRAINING_SECTIONS = {
    "experiment",
    "runtime",
    "distributed",
    "dataset",
    "augmentation",
    "model",
    "source",
    "flow",
    "time_sampling",
    "training",
    "loss",
    "checkpoint",
    "evaluation",
    "wandb",
}


@pytest.mark.parametrize("path", FULL_TRAINING_CONFIGS, ids=lambda path: path.stem)
def test_stage2_and_joint_training_configs_are_self_contained(path):
    raw = yaml.safe_load(path.read_text())
    assert "extends" not in raw
    assert FULL_TRAINING_SECTIONS <= raw.keys()

    config = load_config(path)
    stage, loss_type, _dataset = path.stem.split("_", 2)
    assert config["loss"]["consistency"]["type"] == loss_type
    if stage == "stage2":
        assert config["experiment"]["stage"] == "consistency_distillation"
        assert config["checkpoint"]["init_from"] is not None
    else:
        assert config["experiment"]["stage"] == "joint_training"
        assert config["checkpoint"]["init_from"] is None
        assert config["checkpoint"]["resume"] is None


def test_stage2_psd_from_joint500_config_keeps_warmups_separate():
    config = load_config(PSD_FROM_JOINT_CONFIG)
    assert config["experiment"]["stage"] == "consistency_distillation"
    assert config["training"]["epochs"] == 300
    assert config["training"]["optimizer"]["parameter_groups"] == {
        "model": {"lr": 3.2e-5},
        "source": {"lr": 1.6e-5},
    }
    assert config["training"]["scheduler"]["warmup_epochs"] == 0
    assert config["loss"]["consistency"]["type"] == "psd"
    assert config["loss"]["consistency"]["start_epoch"] == 0
    assert config["loss"]["consistency"]["warmup_epochs"] == 0
    assert config["checkpoint"]["init_from"].endswith(
        "/results/esd/epoch_0500.pt"
    )


def test_yaml_load_and_cli_override():
    config = load_config(CONFIG, ["training.batch_size=2", "runtime.device=cpu"])
    assert config["training"]["batch_size"] == 2
    assert config["runtime"]["device"] == "cpu"
    assert config["flow"]["time_eps"] == 1.0e-5
    assert config["training"]["scheduler"]["step_unit"] == "epoch"
    assert config["training"]["scheduler"]["warmup_start_factor"] == 0.1
    assert config["training"]["max_batches_per_epoch"] is None
    assert config["evaluation"]["interval"] == {"unit": "epoch", "value": None}
    assert config["training"]["checkpoint_interval_steps"] is None
    assert config["loss"]["consistency"]["warmup_steps"] == 0


@pytest.mark.parametrize("path", [JOINT_PSD_CITYSCAPES_CONFIG, DIAGONAL_ADE20K_CONFIG])
def test_void_is_excluded_from_final_predictions_by_default(path):
    config = load_config(path)
    assert config["evaluation"]["exclude_void_from_prediction"] is True


def test_exclude_void_prediction_config_requires_boolean():
    with pytest.raises(
        ValueError, match="evaluation.exclude_void_from_prediction must be a boolean"
    ):
        load_config(CONFIG, ["evaluation.exclude_void_from_prediction=invalid"])


def test_exclude_void_prediction_config_requires_valid_void_index():
    with pytest.raises(ValueError, match="dataset.void_class_index must be a valid"):
        load_config(CONFIG, ["dataset.void_class_index=20"])


def test_prediction_policy_does_not_change_checkpoint_model_signature():
    retained = load_config(
        CONFIG, ["evaluation.exclude_void_from_prediction=false"]
    )
    excluded = load_config(
        CONFIG, ["evaluation.exclude_void_from_prediction=true"]
    )
    assert model_signature(retained) == model_signature(excluded)


def test_joint_psd_cityscapes_main_protocol_is_optimizer_step_based():
    config = load_config(JOINT_PSD_CITYSCAPES_CONFIG)
    assert config["experiment"]["name"] == "dfm_joint_psd_cityscapes_160k"
    assert config["training"]["epochs"] == 1000
    assert config["training"]["max_optimizer_steps"] == 160000
    assert config["training"]["batch_size"] == 16
    assert config["training"]["grad_accum_steps"] == 1
    assert config["training"]["scheduler"] == {
        "name": "poly",
        "warmup_epochs": 0,
        "warmup_start_factor": 1.0e-6,
        "eta_min": 0.0,
        "warmup_steps": 1500,
        "power": 1.0,
        "min_lr": 0.0,
        "step_unit": "optimizer_step",
    }
    assert config["training"]["checkpoint_interval_epochs"] == 0
    assert config["training"]["checkpoint_interval_steps"] == 16000
    assert config["training"]["validation_epochs"] == []
    assert config["loss"]["consistency"]["start"] == {
        "unit": "optimizer_step", "value": 96000,
    }
    assert config["loss"]["consistency"]["warmup_steps"] == 0
    assert config["evaluation"]["interval"] == {
        "unit": "optimizer_step", "value": 16000,
    }
    assert config["dataset"]["image_size"] == [512, 1024]
    assert config["source"]["input_already_normalized"] is True
    assert config["evaluation"]["original_resolution"] is True


def test_scheduler_unit_and_debug_epoch_limit_are_validated():
    with pytest.raises(ValueError, match="step_unit must be epoch"):
        load_config(CONFIG, ["training.scheduler.step_unit=iteration"])
    with pytest.raises(ValueError, match="max_batches_per_epoch"):
        load_config(CONFIG, ["training.max_batches_per_epoch=0"])


@pytest.mark.parametrize("value", ["0", "-1", "true", "1.5"])
def test_gradient_accumulation_requires_positive_integer(value):
    with pytest.raises(ValueError, match="grad_accum_steps must be a positive integer"):
        load_config(CONFIG, [f"training.grad_accum_steps={value}"])


def test_gradient_surgery_accepts_accumulation_two_and_full_recipe_loads():
    path = (
        Path(__file__).parents[1]
        / "configs"
        / "joint_psd_cityscapes_swin_t_adaptive_surgery_fullres_psd_accum2.yaml"
    )
    config = load_config(path)
    assert config["training"]["batch_size"] == 8
    assert config["training"]["grad_accum_steps"] == 2
    assert config["training"]["max_batches_per_epoch"] == 370
    assert config["loss"]["consistency"]["gradient_surgery"]["enabled"] is True
    assert config["loss"]["consistency"]["psd"]["loss_resolution"] == "full"
    assert config["training"]["batch_size"] * config["training"]["grad_accum_steps"] == 16
    assert config["training"]["max_batches_per_epoch"] // config["training"]["grad_accum_steps"] == 185
    assert config["training"]["max_batches_per_epoch"] * config["training"]["batch_size"] == 2960
    assert 185 * 16 == 2960


def test_accum2_full_psd_recipe_differs_from_parent_only_as_intended():
    root = Path(__file__).parents[1]
    parent = load_config(
        root / "configs" / "joint_psd_cityscapes_swin_t_adaptive_surgery_fullres_psd.yaml"
    )
    accumulated = load_config(
        root / "configs"
        / "joint_psd_cityscapes_swin_t_adaptive_surgery_fullres_psd_accum2.yaml"
    )
    normalized = deepcopy(accumulated)
    normalized["experiment"]["name"] = parent["experiment"]["name"]
    normalized["experiment"]["output_dir"] = parent["experiment"]["output_dir"]
    normalized["training"]["batch_size"] = parent["training"]["batch_size"]
    normalized["training"]["grad_accum_steps"] = parent["training"]["grad_accum_steps"]
    normalized["training"]["max_batches_per_epoch"] = parent["training"]["max_batches_per_epoch"]
    normalized["wandb"]["name"] = parent["wandb"]["name"]
    # The loader records the source filename; it is metadata, not an experiment setting.
    normalized["runtime"]["config_path"] = parent["runtime"]["config_path"]
    assert normalized == parent


def test_missing_required_section_has_clear_error(tmp_path):
    raw = yaml.safe_load(CONFIG.read_text())
    del raw["loss"]
    path = tmp_path / "missing.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="Missing required config section: loss"):
        load_config(path)


def test_unknown_key_is_rejected(tmp_path):
    raw = yaml.safe_load(CONFIG.read_text())
    raw["training"]["mystery_option"] = 123
    path = tmp_path / "unknown.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="Unknown config key: training.mystery_option"):
        load_config(path)


def test_unknown_override_is_rejected():
    with pytest.raises(ValueError, match="Unknown override key"):
        load_config(CONFIG, ["training.not_real=1"])


def test_resolved_config_is_saved(tmp_path):
    config = load_config(CONFIG, ["training.batch_size=3"])
    destination = tmp_path / "config_resolved.yaml"
    save_resolved_config(config, destination)
    loaded = yaml.safe_load(destination.read_text())
    assert loaded["training"]["batch_size"] == 3
    assert loaded["runtime"]["config_path"] == str(CONFIG.resolve())


def test_init_from_and_resume_are_mutually_exclusive(tmp_path):
    raw = yaml.safe_load(CONFIG.read_text())
    raw["checkpoint"]["init_from"] = "a.pt"
    raw["checkpoint"]["resume"] = "b.pt"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="mutually exclusive"):
        load_config(path)


def test_precision_validation_rejects_fake_psd_jvp_and_bf16_numerics():
    with pytest.raises(ValueError, match="PSD does not use JVP"):
        load_config(
            Path(__file__).parents[1] / "configs" / "debug_ddp_stage2_psd.yaml",
            ["loss.consistency.precision.jvp_dtype=bf16"],
        )
    with pytest.raises(ValueError, match="numerical_dtype must be fp32"):
        load_config(
            Path(__file__).parents[1] / "configs" / "debug_ddp_stage2_ecld.yaml",
            ["loss.consistency.precision.numerical_dtype=bf16"],
        )


def test_bf16_jvp_requires_runtime_bf16_amp():
    with pytest.raises(ValueError, match="bf16 JVP requires"):
        load_config(
            Path(__file__).parents[1] / "configs" / "debug_ddp_stage2_esd.yaml",
            ["runtime.amp=false"],
        )


def test_legacy_single_gpu_yaml_can_omit_distributed(tmp_path):
    raw = yaml.safe_load(CONFIG.read_text())
    del raw["distributed"]
    path = tmp_path / "legacy.yaml"
    path.write_text(yaml.safe_dump(raw))
    config = load_config(path)
    assert config["distributed"]["enabled"] == "auto"
    assert config["distributed"]["backend"] == "nccl"


def test_legacy_esd_yaml_without_metadata_uses_defaults_and_resolves(tmp_path):
    raw = yaml.safe_load(ESD_CONFIG.read_text())
    raw["loss"]["consistency"].pop("esd")
    legacy_path = tmp_path / "legacy_esd.yaml"
    legacy_path.write_text(yaml.safe_dump(raw))
    config = load_config(legacy_path)
    assert config["loss"]["consistency"]["esd"] == EXPECTED_ESD_METADATA

    resolved_path = tmp_path / "resolved.yaml"
    save_resolved_config(config, resolved_path)
    resolved = yaml.safe_load(resolved_path.read_text())
    assert resolved["loss"]["consistency"]["esd"] == EXPECTED_ESD_METADATA


def test_esd_metadata_rejects_unimplemented_formulation_and_wrong_source():
    with pytest.raises(ValueError, match="formulation must be"):
        load_config(
            ESD_CONFIG,
            ["loss.consistency.esd.formulation=raw"],
        )
    with pytest.raises(ValueError, match="source must be"):
        load_config(
            ESD_CONFIG,
            ["loss.consistency.esd.source=other"],
        )


def test_esd_startup_log_uses_resolved_metadata_and_safeguard_settings():
    config = load_config(ESD_CONFIG)

    class CapturingLogger:
        def __init__(self):
            self.messages = []

        def info(self, message, *args):
            self.messages.append(message % args if args else message)

    logger = CapturingLogger()
    log_esd_experiment_metadata(config, logger)
    assert logger.messages == [
        "ESD formulation: stabilized_logit_space",
        "ESD source: discrete_flow_maps",
        "ESD additional numerical safeguards: true",
        "ESD invalid teacher strategy: mask_pixel",
        "ESD JVP dtype: bf16",
        "ESD numerical dtype: fp32",
    ]
