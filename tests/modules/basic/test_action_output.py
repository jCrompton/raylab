# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch

from raylab.modules.basic import ActionOutput


@pytest.fixture(params=(True, False))
def torch_script(request):
    return request.param


@pytest.fixture
def args_kwargs():
    return (10, torch.ones(4).neg(), torch.ones(4)), dict(beta=1.2)


def test_module_creation(torch_script, args_kwargs):
    maker = ActionOutput.as_script_module if torch_script else ActionOutput
    args, kwargs = args_kwargs
    module = maker(*args, **kwargs)

    inputs = torch.randn_like(args[1])
    module(inputs)
