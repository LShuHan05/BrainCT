#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import re
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pydicom
from tqdm import tqdm

# ----- 可配置路径 -----
DATA_ROOT = Path("datasets_filtered/CT")
ANNOTATION_FILE = Path("md/reads.csv")
OUTPUT_CSV = Path("metadata/dataset_metadata.csv")
EXPORT_NIFTI = True
NIFTI_OUTPUT_DIR = Path("datasets_nifti")

LABEL_COLUMNS = [
    'ICH', 'IPH', 'IVH', 'SDH', 'EDH', 'SAH',
    'Fracture', 'MassEffect', 'MidlineShift'
]
READER_PREFIXES = ['R1:', 'R2:', 'R3:']
# -------------------------------------------------

def normalize_name(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9]', '', str(name)).lower()

def parse_dicom_series(dicom_dir: Path) -> Tuple[List[Path], dict]:
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
            spacing = ds.PixelSpacing if hasattr(ds, 'PixelSpacing') else (1.0, 1.0)
            thickness = ds.SliceThickness if hasattr(ds, 'SliceThickness') else 1.0
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
        'series_uid': getattr(first_ds, 'SeriesInstanceUID', 'Unknown'),
        'modality': getattr(first_ds, 'Modality', 'CT'),
    }
    return sorted_files, metadata

def score_sequence(seq_path: Path) -> float:
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
        dcm_files = list(seq_dir.glob("*.dcm"))
        if not dcm_files:
            continue
        score = score_sequence(seq_dir)
        if score > best_score:
            best_score = score
            best_path = seq_dir
    return best_path

def extract_labels_from_row(row: pd.Series) -> np.ndarray:
    """从标注行提取9个标签，自动转换类型"""
    labels = np.zeros(len(LABEL_COLUMNS), dtype=np.float32)
    for i, col in enumerate(LABEL_COLUMNS):
        max_val = 0
        # 先尝试直接列
        if col in row.index:
            val = row[col]
            try:
                if float(val) > 0:
                    labels[i] = 1.0
                    continue
            except:
                pass
        # 尝试R1/R2/R3
        for prefix in READER_PREFIXES:
            rcol = f"{prefix}{col}"
            if rcol in row.index:
                val = row[rcol]
                try:
                    num = float(val)
                    if num > max_val:
                        max_val = num
                except:
                    continue
        if max_val > 0:
            labels[i] = 1.0
    return labels

def save_as_nifti(sorted_files: List[Path], output_nifti: Path, metadata: dict):
    try:
        import nibabel as nib
        import numpy as np
        volume = []
        for f in sorted_files:
            ds = pydicom.dcmread(f)
            arr = ds.pixel_array.astype(np.float32)
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                arr = arr * ds.RescaleSlope + ds.RescaleIntercept
            volume.append(arr)
        volume = np.stack(volume, axis=0)
        spacing = metadata['spacing']
        affine = np.eye(4)
        affine[0,0] = spacing[0]
        affine[1,1] = spacing[1]
        affine[2,2] = spacing[2]
        img = nib.Nifti1Image(volume, affine)
        nib.save(img, str(output_nifti))
        return True
    except ImportError:
        return False

def main():
    if not ANNOTATION_FILE.exists():
        print(f"❌ 标注文件不存在: {ANNOTATION_FILE}")
        sys.exit(1)
    annotation_df = pd.read_csv(ANNOTATION_FILE)
    annotation_df['name_norm'] = annotation_df['name'].apply(normalize_name)
    print(f"📊 加载标注数据: {len(annotation_df)} 条记录")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    if EXPORT_NIFTI:
        NIFTI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    case_folders = [d for d in DATA_ROOT.iterdir() if d.is_dir()]
    print(f"📁 找到 {len(case_folders)} 个病例文件夹")

    records = []
    debug_printed = False  # 只打印第一个病例的标签

    for case_folder in tqdm(case_folders, desc="处理病例"):
        raw_folder_name = case_folder.name
        raw_id = raw_folder_name.split()[0]
        norm_folder = normalize_name(raw_id)

        matched_rows = annotation_df[annotation_df['name_norm'] == norm_folder]
        if matched_rows.empty:
            continue

        matched_row = matched_rows.iloc[0]
        matched_id = matched_row['name']

        labels = extract_labels_from_row(matched_row)

        # 调试：打印第一个匹配病例的标签
        if not debug_printed:
            print(f"\n调试示例: {matched_id} -> labels = {labels}")
            debug_printed = True

        # 如果所有标签为0，跳过该病例（可选，但这里保留）
        if labels.sum() == 0:
            # 可能真的没有阳性，不跳过，继续处理
            pass

        best_seq = find_best_series(case_folder)
        if best_seq is None:
            continue

        sorted_files, metadata = parse_dicom_series(best_seq)
        if not sorted_files:
            continue

        record = {
            'case_id': matched_id,
            'folder_name': raw_folder_name,
            'sequence_path': str(best_seq.relative_to(DATA_ROOT)),
            'num_slices': metadata['num_slices'],
            'shape': str(metadata['shape']),
            'spacing': str(metadata['spacing']),
            **{f'label_{i}': labels[i] for i in range(len(LABEL_COLUMNS))},
            'sorted_files': ';'.join([str(f.relative_to(DATA_ROOT)) for f in sorted_files])
        }

        if EXPORT_NIFTI:
            nifti_path = NIFTI_OUTPUT_DIR / f"{matched_id}.nii.gz"
            success = save_as_nifti(sorted_files, nifti_path, metadata)
            record['nifti_path'] = str(nifti_path) if success else ''

        records.append(record)

    if records:
        df = pd.DataFrame(records)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"✅ 元数据已保存至: {OUTPUT_CSV}")
        print(f"📊 总计处理 {len(records)} 个病例")

        labels_df = df[[f'label_{i}' for i in range(len(LABEL_COLUMNS))]]
        print("\n📈 各标签阳性样本数:")
        for i, col in enumerate(LABEL_COLUMNS):
            count = labels_df[f'label_{i}'].sum()
            print(f"  {col}: {int(count)}")
    else:
        print("⚠️ 未找到任何匹配的病例")

if __name__ == "__main__":
    main()