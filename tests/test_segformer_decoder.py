from copy import deepcopy
from pathlib import Path

import pytest
import torch

from checkpoint import (
    checkpoint_payload,
    model_signature,
    validate_source_decoder_checkpoint,
)
from config import load_config, validate_config
from source_model import SegFormerSourceGenerator


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "debug" / "diagonal" / "cityscapes.yaml"
STANDARD_CONFIG = (
    ROOT
    / "configs"
    / "cityscapes"
    / "diagonal"
    / "source_segformer_b1_standard_32k.yaml"
)


def _source(*, learned_logvar=False, freeze_encoder=False, decoder="standard"):
    return SegFormerSourceGenerator(
        num_classes=20,
        variant="b1",
        pretrained=False,
        decoder_channels=256,
        freeze_encoder=freeze_encoder,
        learned_logvar=learned_logvar,
        fixed_std=None if learned_logvar else 0.5,
        mu_tanh_scale=0.0,
        input_already_normalized=False,
        state_downsample_factor=4,
        decoder_type=decoder,
    )


@pytest.mark.parametrize("decoder", ["custom", "standard"])
def test_segformer_decoder_config_accepts_supported_values(decoder):
    config = load_config(CONFIG, [f"source.segformer_decoder={decoder}"])
    assert config["source"]["segformer_decoder"] == decoder


@pytest.mark.parametrize("decoder", ["foo", "segformer", "standard_head"])
def test_segformer_decoder_config_rejects_invalid_values(decoder):
    with pytest.raises(ValueError, match="segformer_decoder must be custom or standard"):
        load_config(CONFIG, [f"source.segformer_decoder={decoder}"])


def test_unspecified_segformer_decoder_defaults_to_custom():
    config = load_config(CONFIG)
    assert config["source"]["segformer_decoder"] == "custom"
    assert "segformer_decoder" not in model_signature(config)["source"]
    module = torch.nn.Linear(1, 1)
    payload = checkpoint_payload(
        config=config,
        epoch=0,
        global_step=0,
        model=module,
        source_model=module,
        optimizer=None,
        scheduler=None,
        scaler=None,
        metrics={},
    )
    assert payload["config"]["source"]["segformer_decoder"] == "custom"


def test_b1_standard_stage1_config_inherits_existing_recipe():
    config = load_config(STANDARD_CONFIG)
    assert config["source"]["segformer_variant"] == "b1"
    assert config["source"]["segformer_decoder"] == "standard"
    assert config["source"]["pretrained"] is True
    assert config["source"]["freeze"] is False
    assert config["source"]["freeze_encoder"] is False
    assert config["training"]["train_endpoint"] is False
    assert config["training"]["max_optimizer_steps"] == 32000
    assert config["source"]["supervision"] == {
        "type": "cross_entropy", "weight": 1.0, "include_void": True,
    }


def test_b1_standard_fixed_std_forward_shape_and_constant():
    from transformers.models.segformer.modeling_segformer import SegformerDecodeHead

    source = _source()
    image = torch.rand(1, 3, 64, 128)
    source.eval()
    with torch.no_grad():
        mu, logvar = source.forward_statistics(image)
    assert mu.shape == logvar.shape == (1, 20, 16, 32)
    torch.testing.assert_close(
        logvar, torch.full_like(logvar, torch.log(torch.tensor(0.5**2)))
    )
    assert source.decode_head.config.decoder_hidden_size == 256
    assert source.decode_head.classifier.out_channels == 20
    assert isinstance(source.decode_head, SegformerDecodeHead)

    source.state_downsample_factor = 8
    with torch.no_grad():
        mu_factor8, logvar_factor8 = source.forward_statistics(image)
    assert mu_factor8.shape == logvar_factor8.shape == (1, 20, 8, 16)


def test_standard_decoder_hidden_sizes_match_segformer_variants():
    assert SegFormerSourceGenerator.DECODER_HIDDEN == {
        "b0": 256,
        "b1": 256,
        "b2": 768,
        "b3": 768,
        "b4": 768,
        "b5": 768,
    }


def test_b1_standard_learned_logvar_has_two_twenty_channel_outputs():
    source = _source(learned_logvar=True)
    source.eval()
    with torch.no_grad():
        mu, logvar = source.forward_statistics(torch.rand(1, 3, 64, 128))
    assert mu.shape == logvar.shape == (1, 20, 16, 32)
    assert source.decode_head.classifier.out_channels == 40


@pytest.mark.parametrize("freeze_encoder", [False, True])
def test_standard_head_gradient_and_encoder_freezing(freeze_encoder):
    source = _source(freeze_encoder=freeze_encoder)
    source.train()
    mu, logvar = source.forward_statistics(torch.rand(1, 3, 64, 128))
    (mu.square().mean() + logvar.mean()).backward()
    encoder_gradients = [parameter.grad for parameter in source.encoder.parameters()]
    head_gradients = [parameter.grad for parameter in source.decode_head.parameters()]
    if freeze_encoder:
        assert all(gradient is None for gradient in encoder_gradients)
    else:
        assert any(gradient is not None for gradient in encoder_gradients)
    assert any(gradient is not None for gradient in head_gradients)


def test_custom_mode_keeps_legacy_parameter_names_and_forward_shape():
    source = _source(decoder="custom")
    keys = source.state_dict()
    assert any(key.startswith("projections.") for key in keys)
    assert any(key.startswith("decoder.") for key in keys)
    assert not any(key.startswith("decode_head.") for key in keys)
    source.eval()
    with torch.no_grad():
        mu, logvar = source.forward_statistics(torch.rand(1, 3, 64, 128))
    assert mu.shape == logvar.shape == (1, 20, 16, 32)


def test_checkpoint_decoder_mismatch_is_explicit_and_legacy_means_custom():
    custom = load_config(CONFIG)
    custom["source"]["backbone"] = "segformer"
    standard = deepcopy(custom)
    standard["source"]["segformer_decoder"] = "standard"
    validate_config(standard)
    checkpoint = {"config": custom}
    with pytest.raises(RuntimeError, match="source decoder mismatch"):
        validate_source_decoder_checkpoint(checkpoint, standard, "checkpoint.pt")
    with pytest.raises(RuntimeError, match="source decoder mismatch"):
        validate_source_decoder_checkpoint({"config": standard}, custom, "checkpoint.pt")
    legacy = deepcopy(custom)
    del legacy["source"]["segformer_decoder"]
    validate_source_decoder_checkpoint({"config": legacy}, custom, "legacy.pt")
