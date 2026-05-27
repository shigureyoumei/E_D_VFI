import importlib
from os import path as osp

from basicsr.utils import scandir

# REFID baseline architectures live under refid/, while custom Mamba
# architectures live under mamba/. Shared building blocks stay in archs/.
arch_folder = osp.dirname(osp.abspath(__file__))
arch_filenames = [
    osp.splitext(osp.basename(v))[0] for v in scandir(arch_folder)
    if v.endswith('_arch.py')
]
arch_filenames.extend([
    f'refid.{osp.splitext(osp.basename(v))[0]}'
    for v in scandir(osp.join(arch_folder, 'refid'))
    if v.endswith('_arch.py')
])
# Import baseline architecture modules eagerly; Mamba modules pull optional
# CUDA/Triton dependencies and are loaded only when their network is requested.
_arch_modules = [
    importlib.import_module(f'basicsr.models.archs.{file_name}')
    for file_name in arch_filenames
]


def find_arch_class(modules, cls_type):
    """Find an architecture class in imported modules.

    Args:
        modules (list[importlib modules]): List of modules from importlib
            files.
        cls_type (str): Class type.

    Returns:
        class | None: Located class object.
    """
    for module in modules:
        cls_ = getattr(module, cls_type, None)
        if cls_ is not None:
            return cls_
    return None


def define_network(opt):
    network_type = opt.pop('type')
    cls_ = find_arch_class(_arch_modules, network_type)
    if cls_ is None:
        mamba_module = importlib.import_module(
            'basicsr.models.archs.mamba.MambaMotionBidirectionalNetwork')
        cls_ = find_arch_class([mamba_module], network_type)
    if cls_ is None:
        raise ValueError(f'{network_type} is not found.')
    return cls_(**opt)
