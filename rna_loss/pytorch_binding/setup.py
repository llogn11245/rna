import io
import os

import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_requirements():
    req_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    with io.open(req_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def get_long_description():
    readme_file = os.path.join(os.path.dirname(__file__), "README.md")
    with io.open(readme_file, "r", encoding="utf-8") as f:
        return f.read()


if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required. CPU version is not implemented.")

requirements = get_requirements()
long_description = get_long_description()

setup(
    name="warp_rna",
    version="0.3.0",
    description="PyTorch bindings for CUDA-Warp Recurrent Neural Aligner",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/1ytic/warp-rna/tree/master/pytorch_binding",
    author="Ivan Sorokin",
    author_email="sorokin.ivan@inbox.ru",
    license="MIT",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="warp_rna._C",
            sources=["./binding.cpp", "./core.cu"],
            include_dirs=[
                os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            ] + torch.utils.cpp_extension.include_paths()
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=requirements,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ]
)
