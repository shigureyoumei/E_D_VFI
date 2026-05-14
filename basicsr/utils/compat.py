import inspect

import torch


def torch_load_legacy_pickle(path, *args, **kwargs):
    """Load trusted legacy checkpoints across PyTorch versions.

    PyTorch 2.6 changed torch.load's default weights_only behavior. The
    checkpoints and training states used by this project are trusted local
    artifacts and may include optimizer/scheduler pickle payloads, so keep the
    pre-2.6 behavior when the runtime supports the argument.
    """
    try:
        load_params = inspect.signature(torch.load).parameters
    except (TypeError, ValueError):
        load_params = {}
    if 'weights_only' in load_params:
        kwargs.setdefault('weights_only', False)
    return torch.load(path, *args, **kwargs)
