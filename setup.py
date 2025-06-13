from setuptools import setup, find_packages

setup(
    name="ultrasegmentator",  # PyPI 上的包名（必须唯一）
    py_modules=["ultrasegmentator"],  # 模块名
    version="0.1.0",  # 版本号
    author="weki",
    python_requires="==3.10.*",  # 严格限定Python 3.10
    install_requires=[
        "torch==2.6.*",  # 限定torch 2.6系列
        "numpy==2.1.*",  # 限定numpy 2.1系列
        "cuda-python>=12.1.0",  # CUDA Python绑定
    ],
    extras_require={
        "cuda": ["nvidia-cudnn-cu12==8.9.*"]  # CUDA 12相关依赖
    },
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",  # 通常CUDA只在Linux支持
        "Environment :: GPU :: NVIDIA CUDA :: 12.1+",
    ]
)