from __future__ import annotations
import multiprocessing
import os
from typing import List
from pathlib import Path
from warnings import warn

import numpy as np
from tqdm import tqdm
from batchgenerators.utilities.file_and_folder_operations import isfile, subfiles
from ultrasegmentator.configuration import default_num_processes

def _convert_to_npy(args) -> None:
    npz_file, unpack_segmentation, overwrite_existing, verify_npy, fail_ctr = args
    data_npy = npz_file[:-3] + "npy"
    seg_npy = npz_file[:-4] + "_seg.npy"
    try:
        npz_content = None  # will only be opened on demand

        if overwrite_existing or not isfile(data_npy):
            try:
                npz_content = np.load(npz_file) if npz_content is None else npz_content
            except Exception as e:
                print(f"Unable to open preprocessed file {npz_file}. Rerun nnUNetv2_preprocess!")
                raise e
            np.save(data_npy, npz_content['data'])

        if unpack_segmentation and (overwrite_existing or not isfile(seg_npy)):
            try:
                npz_content = np.load(npz_file) if npz_content is None else npz_content
            except Exception as e:
                print(f"Unable to open preprocessed file {npz_file}. Rerun nnUNetv2_preprocess!")
                raise e
            np.save(npz_file[:-4] + "_seg.npy", npz_content['seg'])

        if verify_npy:
            try:
                print(data_npy)
                np.load(data_npy, mmap_mode='r')
                
                if isfile(seg_npy):
                    np.load(seg_npy, mmap_mode='r')
            except ValueError:
                
                os.remove(data_npy)
                os.remove(seg_npy)
                print(f"Error when checking {data_npy} and {seg_npy}, fixing...")
                if fail_ctr < 2:
                    _convert_to_npy((npz_file, unpack_segmentation, overwrite_existing, verify_npy, fail_ctr+1))
                else:
                    raise RuntimeError("Unable to fix unpacking. Please check your system or rerun nnUNetv2_preprocess")

        # # 删除已成功解压的 npz 文件
        # os.remove(npz_file)

    except KeyboardInterrupt:
        if isfile(data_npy):
            os.remove(data_npy)
        if isfile(seg_npy):
            os.remove(seg_npy)
        raise KeyboardInterrupt

def unpack_dataset(folder: str, unpack_segmentation: bool = True, overwrite_existing: bool = False,
                   num_processes: int = default_num_processes, verify_npy: bool = False):
    """
    all npz files in this folder belong to the dataset, unpack them all
    """
    npz_files = subfiles(folder, True, None, ".npz", True)
    
    # 初始化 tqdm 进度条
    with tqdm(total=len(npz_files), desc="Unpacking files") as progress_bar:
        with multiprocessing.get_context("spawn").Pool(num_processes) as pool:
            for _ in pool.imap_unordered(_convert_to_npy, [(npz_file, unpack_segmentation, overwrite_existing, verify_npy, 0) for npz_file in npz_files]):
                progress_bar.update()

def get_case_identifiers(folder: str) -> List[str]:
    """
    finds all npz files in the given folder and reconstructs the training case names from them
    """
    case_identifiers = [i[:-4] for i in os.listdir(folder) if i.endswith("npz") and (i.find("segFromPrevStage") == -1)]
    return case_identifiers


if __name__ == '__main__':
    unpack_dataset('/mnt/data1/yujunxuan/MICCAI_challenge_method/nnUNet-master/ultrasegmentator/Dataset/nnUNet_preprocessed/Dataset130_MICCAI_challenge/nnUNetPlans_3d_fullres')
