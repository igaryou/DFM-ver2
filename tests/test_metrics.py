import torch

from metrics import SegmentationMetrics


def test_gt_void_is_excluded_but_predicted_void_is_an_error():
    metrics = SegmentationMetrics()
    target = torch.tensor([[0, 0, 19]])
    prediction = torch.tensor([[0, 19, 0]])
    metrics.update(prediction, target)
    result = metrics.compute()
    assert result["confusion_matrix"][0][0] == 1
    assert result["confusion_matrix"][0][19] == 1
    assert sum(result["confusion_matrix"][19]) == 0
    assert result["pixel_acc"] == 0.5
    assert result["prediction_void_retained"] is True

