'''
Author: your name
Date: 2021-03-16 15:19:46
LastEditTime: 2021-03-16 19:30:54
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Demo/extension/mish/setup.py
'''
import os
import glob
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension
from torch.utils.cpp_extension import BuildExtension


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "src")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

    sources = main_file + source_cpu
    extension = CppExtension
    extra_compile_args = {"cxx": []}
    define_macros = []

    if torch.cuda.is_available() and CUDA_HOME is not None:
        extension = CUDAExtension
        sources += source_cuda
        # define_macros += [("WITH_CUDA", None)]
        extra_compile_args['nvcc'] = ['--expt-extended-lambda']
    else:
        raise NotImplementedError('Cuda is not availabel')

    sources = [os.path.join(extensions_dir, s) for s in sources]
    include_dirs = [extensions_dir, "/usr/local/cuda-11.0/include"]
    ext_modules = [
        extension(
            "Mish",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]
    return ext_modules


setup(name='Mish',
      version='1.0',
      author="qinyi",
      packages=find_packages(),
      include_package_data=True,
      ext_modules=get_extensions(),
      cmdclass={'build_ext': BuildExtension})
