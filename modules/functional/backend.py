import os

from torch.utils.cpp_extension import load

_src_path = os.path.dirname(os.path.abspath(__file__))
_backend = load(name='_pvcnn_backend',
                extra_cflags=['-O3', '-std=c++17'],
                extra_cuda_cflags=['--compiler-bindir=/usr/bin/gcc'],
                sources=[os.path.join(_src_path,'src', f) for f in [
                    'ball_query/ball_query.cpp',
                    'ball_query/ball_query.cu',
                    'grouping/grouping.cpp',
                    'grouping/grouping.cu',
                    'interpolate/neighbor_interpolate.cpp',
                    'interpolate/neighbor_interpolate.cu',
                    'interpolate/trilinear_devox.cpp',
                    'interpolate/trilinear_devox.cu',
                    'sampling/sampling.cpp',
                    'sampling/sampling.cu',
                    'voxelization/vox.cpp',
                    'voxelization/vox.cu',
                    'bindings.cpp',
                ]]
                )

__all__ = ['_backend']

# import os
# import torch
# from torch.utils.cpp_extension import load
#
# # 获取当前脚本的路径
# _src_path = os.path.dirname(os.path.abspath(__file__))
#
# # 编译 C++ 和 CUDA 扩展模块
# _backend = load(name='_pvcnn_backend',
#                 extra_cflags=['/O2', '/std:c++17'],  # 使用 Windows 上的编译选项
#                 extra_cuda_cflags=[],  # 不需要 --compiler-bindir 参数
#                 sources=[os.path.join(_src_path, 'src', f) for f in [
#                     'ball_query/ball_query.cpp',
#                     'ball_query/ball_query.cu',
#                     'grouping/grouping.cpp',
#                     'grouping/grouping.cu',
#                     'interpolate/neighbor_interpolate.cpp',
#                     'interpolate/neighbor_interpolate.cu',
#                     'interpolate/trilinear_devox.cpp',
#                     'interpolate/trilinear_devox.cu',
#                     'sampling/sampling.cpp',
#                     'sampling/sampling.cu',
#                     'voxelization/vox.cpp',
#                     'voxelization/vox.cu',
#                     'bindings.cpp',
#                 ]])
#
# # 导出 _backend 模块
# __all__ = ['_backend']
#
