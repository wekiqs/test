import sys
import os
import urllib

from pathlib import Path
from typing import Union
from ultrasegmentator.paths import nnUNet_results, nnUNet_raw
from ultrasegmentator.inference.predict_from_raw_data import *

import numpy as np
import torch
import re


def validate_device_type_api(value):
    valid_strings = ["gpu", "cpu", "mps"]
    if value in valid_strings:
        return value

    # Check if the value matches the pattern "gpu:X" where X is an integer
    pattern = r"^gpu:(\d+)$"
    match = re.match(pattern, value)
    if match:
        device_id = int(match.group(1))
        return value

    raise ValueError(
        f"Invalid device type: '{value}'. Must be 'gpu', 'cpu', 'mps', or 'gpu:X' where X is an integer representing the GPU device ID.")


def convert_device_to_cuda(device):
    if device in ["cpu", "mps", "gpu"]:
        return device
    else:  # gpu:X
        return f"cuda:{device.split(':')[1]}"


def get_cache_dir() -> Path:
    """
    获取系统缓存目录路径（跨平台）
    Windows: %LOCALAPPDATA%\\Temp
    Linux: /var/tmp 或 /tmp
    """
    if sys.platform.startswith('win'):
        # Windows 缓存路径
        cache_base = os.environ.get('LOCALAPPDATA', os.path.expanduser('~'))
        return Path(cache_base) / 'Temp'
    else:
        # Linux/Unix 缓存路径
        for path in ('/var/tmp', '/tmp'):
            if os.path.isdir(path):
                return Path(path)
        # 回退到用户目录
        return Path.home() / '.cache'


def download_file(url: str, file_path) -> Path:
    """
    下载文件到缓存目录（如果不存在）
    返回下载文件的完整路径
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # 检查文件是否已存在
    if os.path.exists(file_path):
        print(f"文件已存在，跳过下载: {file_path}")
        return file_path

    print(f"开始下载: {url}")
    try:
        # 下载文件
        urllib.request.urlretrieve(url, file_path)
        print(f"文件已保存到: {file_path}")
        return file_path
    except Exception as e:
        # 清理可能创建的空文件
        if os.path.exists(file_path):
            os.remove(file_path)
        raise RuntimeError(f"下载失败: {e}") from e

def pred(input: Union[str, Path], output: Union[str, Path, None] = None,tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                      perform_everything_on_device=True, device="gpu", verbose=False, quiet=False,save_probabilities =False,
                      skip_saving=False, force_split=False, nr_thr_resamp=1, nr_thr_saving=6,num_parts=1, part_id=0):
    """
    Ultrasegmentator API for nnUNet inference.
    :param input:
    :param output:
    :param tile_step_size:
    :param use_gaussian:
    :param use_mirroring:
    :param perform_everything_on_device:
    :param device:
    :param verbose:
    :param quiet:
    :param skip_saving:
    :param force_split:
    :param nr_thr_resamp:
    :param nr_thr_saving:
    :return:
    """
    # 处理设备参数
    device = torch.device('cuda') if device == "gpu" else torch.device('cpu')

    # 初始化预测器
    predictor = nnUNetPredictor(
        tile_step_size=tile_step_size,
        use_gaussian=use_gaussian,
        use_mirroring=use_mirroring,
        perform_everything_on_device=perform_everything_on_device,
        device=device,
        verbose=verbose,  # 使用用户指定的verbose参数
        verbose_preprocessing=verbose,
        allow_tqdm=not quiet  # quiet模式关闭tqdm进度条
    )

    cache_dir = get_cache_dir()
    # 检查模型路径是否存在，如果不存在则下载
    model_url = "https://github.com/wekiqs/test/releases/download/nnunet/nnunetv1.pth"  # 替换为实际的模型下载链接
    download_file(model_url, os.path.join(cache_dir, "nnUNet.pth"))
    plan_url = "https://github.com/wekiqs/test/releases/download/nnunet/plans.json"
    download_file(plan_url, os.path.join(cache_dir, "plans.json"))
    data_url = "https://github.com/wekiqs/test/releases/download/nnunet/dataset.json"
    download_file(data_url, os.path.join(cache_dir, "dataset.json"))

    # 处理输出路径
    if skip_saving:
        output = None
    else:
        if output is None and not os.path.exists(output):
            output.mkdir(parents=True, exist_ok=True)

    # 加载预训练模型
    predictor.initialize_from_path(str(cache_dir))

    # 核心预测调用
    predictor.predict_from_files(
        str(input),
        str(output) if output else None,
        save_probabilities=save_probabilities,
        overwrite=force_split,  # 映射force_split到覆盖选项
        num_processes_preprocessing=nr_thr_resamp,
        num_processes_segmentation_export=nr_thr_saving,
        folder_with_segs_from_prev_stage=None,
        num_parts=num_parts,
        part_id=part_id
    )
