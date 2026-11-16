import torch

from dcfg.utils.seed import seed_everything


def test_reproducible():
    seed_everything(123)
    a = torch.rand(5)
    seed_everything(123)
    assert torch.equal(a, torch.rand(5))


def test_deterministic_flags():
    seed_everything(0, deterministic=True)
    assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
