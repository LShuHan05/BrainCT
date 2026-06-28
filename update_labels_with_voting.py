# update_labels_with_voting.py
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Conf.Config import ANNOTATION_FILE, METADATA_CSV, METADATA_DIR

READER_PREFIXES = ['R1:', 'R2:', 'R3:']
LABEL_COLUMNS = [
    'ICH', 'IPH', 'IVH', 'SDH', 'EDH', 'SAH',
    'Fracture', 'MassEffect', 'MidlineShift'
]

def extract_labels_with_voting(row, min_votes=2):
    """提取 9 个标签，要求至少 min_votes 位医生同意"""
    labels = []
    for col in LABEL_COLUMNS:
        # 1. 先检查是否有汇总列（直接取）
        if col in row.index:
            try:
                if float(row[col]) > 0:
                    labels.append(1.0)
                    continue
            except:
                pass
        # 2. 统计三位医生的投票
        votes = 0
        for prefix in READER_PREFIXES:
            rcol = f"{prefix}{col}"
            if rcol in row.index:
                try:
                    val = float(row[rcol])
                    if val > 0:
                        votes += 1
                except:
                    continue
        if votes >= min_votes:
            labels.append(1.0)
        else:
            labels.append(0.0)
    return labels

def main():
    print("=" * 70)
    print("🔄 正在更新标签 (采用 ≥2 位医生投票机制)")
    print("=" * 70)

    # 1. 加载原始标注
    reads_df = pd.read_csv(ANNOTATION_FILE)
    print(f"✅ 加载标注文件: {len(reads_df)} 条")

    # 2. 加载现有的 metadata
    metadata_path = METADATA_CSV
    if not os.path.exists(metadata_path):
        print(f"❌ 未找到 metadata 文件: {metadata_path}")
        return

    metadata_df = pd.read_csv(metadata_path)
    print(f"✅ 加载现有 metadata: {len(metadata_df)} 例")

    # 3. 构建 case_id 映射
    id_to_row = {}
    for _, row in reads_df.iterrows():
        id_to_row[row['name']] = row

    # 4. 更新标签列
    updated_count = 0
    for idx, meta_row in metadata_df.iterrows():
        case_id = meta_row['case_id']
        if case_id not in id_to_row:
            print(f"⚠️  未找到标注: {case_id}，跳过")
            continue
        raw_row = id_to_row[case_id]
        new_labels = extract_labels_with_voting(raw_row, min_votes=2)
        for i in range(len(LABEL_COLUMNS)):
            metadata_df.at[idx, f'label_{i}'] = new_labels[i]
        updated_count += 1

    print(f"✅ 已更新 {updated_count} 个病例的标签")

    # 5. 备份并保存
    backup_path = metadata_path.replace('.csv', '_backup.csv')
    os.rename(metadata_path, backup_path)
    print(f"💾 原文件已备份至: {backup_path}")

    metadata_df.to_csv(metadata_path, index=False)
    print(f"💾 新标签已保存至: {metadata_path}")

    # 6. 打印新分布
    print("\n【新标签分布】")
    for i, col in enumerate(LABEL_COLUMNS):
        cnt = metadata_df[f'label_{i}'].sum()
        print(f"  {col:12s}: {int(cnt):4d} 例 ({cnt/len(metadata_df)*100:.2f}%)")

    print("\n✅ 标签更新完成！请重新运行训练。")
    print("=" * 70)

if __name__ == "__main__":
    main()