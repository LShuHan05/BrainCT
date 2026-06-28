#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从原始 CQ500_orig 目录预处理数据集
输入：CQ500_orig/ 下的病例文件夹（直接包含 Unknown Study/ 等）
输出：datasets_processed/nifti/ + metadata/
"""

import os
import sys
import re
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import pydicom
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Conf.Config import *

# 标签定义
LABEL_COLUMNS = [
    'ICH', 'IPH', 'IVH', 'SDH', 'EDH', 'SAH',
    'Fracture', 'MassEffect', 'MidlineShift'
]
READER_PREFIXES = ['R1:', 'R2:', 'R3:']

def normalize_name(name: str) -> str:
    """标准化名称用于匹配"""
    return re.sub(r'[^a-zA-Z0-9]', '', str(name)).lower()

def parse_dicom_series(dicom_dir: Path) -> Tuple[List[Path], dict]:
    """读取并排序DICOM序列"""
    dcm_files = list(dicom_dir.glob("*.dcm"))
    if not dcm_files:
        return [], {}

    slices = []
    for f in dcm_files:
        try:
            ds = pydicom.dcmread(f, force=True)
            if hasattr(ds, 'ImagePositionPatient'):
                z_pos = float(ds.ImagePositionPatient[2])
            elif hasattr(ds, 'InstanceNumber'):
                z_pos = int(ds.InstanceNumber)
            else:
                z_pos = 0
            slices.append((z_pos, f, ds))
        except Exception:
            continue

    if not slices:
        return [], {}

    slices.sort(key=lambda x: x[0])
    sorted_files = [s[1] for s in slices]
    first_ds = slices[0][2]
    metadata = {
        'num_slices': len(sorted_files),
        'shape': (len(sorted_files), first_ds.Rows, first_ds.Columns),
        'spacing': (float(first_ds.PixelSpacing[0]), float(first_ds.PixelSpacing[1]),
                    float(first_ds.SliceThickness) if hasattr(first_ds, 'SliceThickness') else 1.0),
    }
    return sorted_files, metadata

def score_sequence(seq_path: Path) -> float:
    """序列评分：优先平扫+薄层"""
    path_str = str(seq_path).lower()
    score = 0.0
    if 'plain' in path_str and 'contrast' not in path_str:
        score += 100
    if 'thin' in path_str:
        score += 50
    dcm_files = list(seq_path.glob("*.dcm"))
    if dcm_files:
        try:
            ds = pydicom.dcmread(dcm_files[0], force=True)
            thickness = getattr(ds, 'SliceThickness', 5.0)
            if thickness <= 1.25:
                score += 30
            elif thickness <= 2.0:
                score += 15
            score += min(len(dcm_files) / 50.0, 10.0)
        except:
            pass
    return score

def find_best_series(case_folder: Path) -> Optional[Path]:
    """查找病例的最佳序列"""
    study_folder = case_folder / "Unknown Study"
    if not study_folder.exists():
        subdirs = [d for d in case_folder.iterdir() if d.is_dir()]
        if not subdirs:
            return None
        study_folder = case_folder

    best_path = None
    best_score = -1.0
    for seq_dir in study_folder.iterdir():
        if not seq_dir.is_dir():
            continue
        if not list(seq_dir.glob("*.dcm")):
            continue
        score = score_sequence(seq_dir)
        if score > best_score:
            best_score = score
            best_path = seq_dir
    return best_path

def extract_labels_from_row(row: pd.Series) -> np.ndarray:
    """提取9个标签"""
    labels = np.zeros(len(LABEL_COLUMNS), dtype=np.float32)
    for i, col in enumerate(LABEL_COLUMNS):
        max_val = 0
        if col in row.index:
            try:
                if float(row[col]) > 0:
                    labels[i] = 1.0
                    continue
            except:
                pass
        for prefix in READER_PREFIXES:
            rcol = f"{prefix}{col}"
            if rcol in row.index:
                try:
                    num = float(row[rcol])
                    if num > max_val:
                        max_val = num
                except:
                    continue
        if max_val > 0:
            labels[i] = 1.0
    return labels

# 仅修改 save_as_nifti 函数，其他保持不变

def save_as_nifti(sorted_files: List[Path], output_path: Path, metadata: dict, target_shape=TARGET_SHAPE_3D) -> bool:
    try:
        import nibabel as nib
        from scipy.ndimage import zoom
        volume = []
        for f in sorted_files:
            ds = pydicom.dcmread(f, force=True)  # ← 添加 force=True
            arr = ds.pixel_array.astype(np.float32)
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                arr = arr * ds.RescaleSlope + ds.RescaleIntercept
            volume.append(arr)
        volume = np.stack(volume, axis=0)
        
        # 调试打印（可选，可删除）
        print(f"[DEBUG] {output_path.name}: {volume.shape} -> ", end="")
        
        curr_shape = volume.shape
        zoom_factors = [
            target_shape[0] / curr_shape[0],
            target_shape[1] / curr_shape[1],
            target_shape[2] / curr_shape[2]
        ]
        volume = zoom(volume, zoom_factors, order=1)
        
        print(f"{volume.shape}")
        
        spacing = metadata['spacing']
        affine = np.eye(4)
        affine[0,0] = spacing[0] * (curr_shape[1] / target_shape[1])
        affine[1,1] = spacing[1] * (curr_shape[2] / target_shape[2])
        affine[2,2] = spacing[2] * (curr_shape[0] / target_shape[0])
        
        img = nib.Nifti1Image(volume, affine)
        nib.save(img, str(output_path))
        return True
    except Exception as e:
        print(f"[ERROR] 保存 {output_path.name} 失败: {e}")
        return False

def process_single_case(case_folder: Path, annotation_df: pd.DataFrame, output_dir: Path) -> Optional[dict]:
    """处理单个病例（用于多线程）"""
    try:
        raw_folder_name = case_folder.name
        raw_id = raw_folder_name.split()[0]
        norm_folder = normalize_name(raw_id)

        matched_rows = annotation_df[annotation_df['name_norm'] == norm_folder]
        if matched_rows.empty:
            return None

        matched_row = matched_rows.iloc[0]
        matched_id = matched_row['name']
        labels = extract_labels_from_row(matched_row)

        best_seq = find_best_series(case_folder)
        if best_seq is None:
            return None

        sorted_files, metadata = parse_dicom_series(best_seq)
        if len(sorted_files) < 10:  # 过滤切片太少的序列
            return None

        # 保存NIfTI
        nifti_path = output_dir / "nifti" / f"{matched_id}.nii.gz"
        if not save_as_nifti(sorted_files, nifti_path, metadata):
            return None

        # 构建记录
        record = {
            'case_id': matched_id,
            'folder_name': raw_folder_name,
            'num_slices': metadata['num_slices'],
            'shape': str(metadata['shape']),
            'spacing': str(metadata['spacing']),
            **{f'label_{i}': labels[i] for i in range(len(LABEL_COLUMNS))}
        }
        return record
    except Exception as e:
        print(f"处理 {case_folder.name} 失败: {e}")
        return None

def main():
    start_time = time.time()
    print("=" * 60)
    print("开始预处理 CQ500 原始数据集")
    print(f"输入目录: {CQ500_ORIG_ROOT}")
    print(f"输出目录: {PREPROCESS_OUTPUT_DIR}")
    print("=" * 60)

    # 创建输出目录
    nifti_dir = Path(NIFTI_DIR)
    metadata_dir = Path(METADATA_DIR)
    nifti_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # 加载标注
    annotation_df = pd.read_csv(ANNOTATION_FILE)
    annotation_df['name_norm'] = annotation_df['name'].apply(normalize_name)
    print(f"加载标注: {len(annotation_df)} 条")

    # 获取病例文件夹
    case_folders = [d for d in Path(CQ500_ORIG_ROOT).iterdir() if d.is_dir()]
    print(f"找到 {len(case_folders)} 个病例文件夹")

    # 多线程处理（利用8核CPU）
    num_workers = min(mp.cpu_count(), 8)
    print(f"使用 {num_workers} 个线程并行处理")

    records = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_single_case, folder, annotation_df, Path(PREPROCESS_OUTPUT_DIR)): folder
            for folder in case_folders
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="处理病例"):
            result = future.result()
            if result:
                records.append(result)

    # 保存元数据
    if records:
        df = pd.DataFrame(records)
        df.to_csv(METADATA_CSV, index=False)
        print(f"✅ 元数据已保存: {METADATA_CSV}")
        print(f"📊 成功处理 {len(records)} 个病例")

        # 统计标签分布
        labels_df = df[[f'label_{i}' for i in range(len(LABEL_COLUMNS))]]
        print("\n📈 各标签阳性样本数:")
        for i, col in enumerate(LABEL_COLUMNS):
            print(f"  {col}: {int(labels_df[f'label_{i}'].sum())}")
    else:
        print("⚠️ 未成功处理任何病例")

    elapsed = (time.time() - start_time) / 60
    print(f"\n⏱️ 总耗时: {elapsed:.1f} 分钟")

if __name__ == "__main__":
    main()