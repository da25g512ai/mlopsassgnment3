import torch
from src.model import get_model
def test_model_shape():
    model = get_model("resnet18", 10)
    out = model(torch.randn(1, 3, 32, 32))
    assert out.shape == (1, 10)