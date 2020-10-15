from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='gpu_utils',
    ext_modules=[
        CUDAExtension('gpu_utils', [
            'gpu_utils.cpp',
            'gpu_utils_cuda.cu',
        ],)
        #  extra_compile_args={
            #  'cxx': [],
            #  'nvcc': ['-g','-G']
        #  })
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
)
