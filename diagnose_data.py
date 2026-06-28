# diagnose_data.py
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Conf.Config import ANNOTATION_FILE, METADATA_CSV

READER_PREFIXES = ['R1:', 'R2:', 'R3:']
LABEL_COLUMNS = [
    'ICH', 'IPH', 'IVH', 'SDH', 'EDH', 'SAH',
    'Fracture', 'MassEffect', 'MidlineShift'
]

def extract_label_with_vote(row, col, prefixes, min_votes=1):
    """提取标签：min_votes=1 为原逻辑，min_votes=2 为清洗逻辑"""
    # 尝试直接列
    if col in row.index:
        try:
            if float(row[col]) > 0:
                return 1.0
        except:
            pass
    # 统计三位医生的投票
    votes = 0
    for prefix in prefixes:
        rcol = f"{prefix}{col}"
        if rcol in row.index:
            try:
                val = float(row[rcol])
                if val > 0:
                    votes += 1
            except:
                continue
    if votes >= min_votes:
        return 1.0
    return 0.0

def main():
    print("=" * 70)
    print("📊 CQ500 数据诊断报告")
    print("=" * 70)

    # 1. 加载数据
    reads_df = pd.read_csv(ANNOTATION_FILE)
    metadata_df = pd.read_csv(METADATA_CSV)
    
    print(f"✅ 标注文件: {len(reads_df)} 条记录")
    print(f"✅ 已处理病例: {len(metadata_df)} 例\n")

    # 2. 当前标签分布（原始 max 逻辑）
    print("【当前标签分布 (max / 任一医生阳性)】")
    current_counts = []
    for i, col in enumerate(LABEL_COLUMNS):
        cnt = metadata_df[f'label_{i}'].sum()
        pct = cnt / len(metadata_df) * 100
        current_counts.append(cnt)
        print(f"  {col:12s}: {int(cnt):4d} 例 ({pct:5.2f}%)")
    print()

    # 3. 模拟“至少2位医生同意”的新标签分布
    print("【模拟新标签分布 (≥2 位医生同意)】")
    new_counts = []
    # 构建 case_id 到原始行的映射
    id_to_row = {}
    for _, row in reads_df.iterrows():
        id_to_row[row['name']] = row

    for _, meta_row in metadata_df.iterrows():
        case_id = meta_row['case_id']
        if case_id not in id_to_row:
            continue
        raw_row = id_to_row[case_id]
        for i, col in enumerate(LABEL_COLUMNS):
            new_label = extract_label_with_vote(raw_row, col, READER_PREFIXES, min_votes=2)
            # 把新标签暂存到 metadata_df 临时列（仅用于统计，不保存）
            metadata_df.loc[_, f'temp_label_{i}'] = new_label

    for i, col in enumerate(LABEL_COLUMNS):
        cnt = metadata_df[f'temp_label_{i}'].sum()
        pct = cnt / len(metadata_df) * 100
        new_counts.append(cnt)
        print(f"  {col:12s}: {int(cnt):4d} 例 ({pct:5.2f}%)")
    print()

    # 4. 对比分析（争议样本 = 原阳性 - 新阳性）
    print("【争议样本统计 (仅1位医生认为阳性)】")
    total_disputed = 0
    for i, col in enumerate(LABEL_COLUMNS):
        disputed = current_counts[i] - new_counts[i]
        total_disputed += disputed
        if disputed > 0:
            print(f"  {col:12s}: {int(disputed):4d} 例 (占原阳性 {disputed/current_counts[i]*100:.1f}%)")
    print(f"  总计争议样本: {int(total_disputed)} 次标注")
    print()

    # 5. 全阴性样本统计（前6类出血）
    bleed_cols = [f'label_{i}' for i in range(6)]
    metadata_df['has_bleed'] = metadata_df[bleed_cols].sum(axis=1) > 0
    no_bleed_pct = (1 - metadata_df['has_bleed'].mean()) * 100
    print(f"【全阴性样本 (无任何出血)】")
    print(f"  完全无出血病例: {no_bleed_pct:.2f}% ({int((~metadata_df['has_bleed']).sum())} 例)")
    print()

    # 6. 结论建议
    print("【诊断结论】")
    if no_bleed_pct > 50:
        print("  ⚠️  超过半数病例完全正常，模型倾向于全阴性预测，严重拉低 F1。")
    if total_disputed > 100:
        print(f"  ⚠️  存在 {int(total_disputed)} 个争议标注（仅1位医生认定），需清洗标签。")
    if max(new_counts) < 30 and min(new_counts) < 10:
        print("  ⚠️  稀有类样本极少（<30），建议合并为二分类或采用过采样。")
    print("\n✅ 诊断完成。")
    print("=" * 70)

if __name__ == "__main__":
    main()