# https://github.dev/mit-han-lab/pvcnn/blob/master/modules/functional/backend.py

import os
from functools import lru_cache


@lru_cache(maxsize=1)
def get_backend():
    # Compile the extension only when a CUDA tensor actually uses it.
    from torch.utils.cpp_extension import load

    src_path = os.path.dirname(os.path.abspath(__file__))
    return load(
        name='_pvcnn_backend',
        extra_cflags=['-O3', '-std=c++17'],
        sources=[os.path.join(src_path, 'src', filename) for filename in [
            'voxelization/vox.cpp',
            'voxelization/vox.cu',
            'interpolate/trilinear_devox.cpp',
            'interpolate/trilinear_devox.cu',
            'bindings.cpp',
        ]],
    )


__all__ = ['get_backend']
