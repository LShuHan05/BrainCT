import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
import requests
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import zipfile
from pathlib import Path
import tempfile
import plotly.graph_objects as go

# FastAPI 服务地址
API_URL = "http://127.0.0.1:8000"
TEMP_DIR = tempfile.gettempdir()

# ==================== 辅助函数 ====================
def load_dicom_from_zip(zip_path):
    """
    从ZIP文件中读取DICOM，按SeriesInstanceUID分组，选择切片数最多的序列，
    并按ImagePositionPatient排序，返回3D体积（D, H, W）。
    """
    import numpy as np
    import pydicom
    import tempfile
    import zipfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        dcm_files = list(Path(tmpdir).glob("*.dcm"))
        if not dcm_files:
            return None

        # 按 SeriesInstanceUID 分组
        series_groups = {}
        for dcm_path in dcm_files:
            try:
                ds = pydicom.dcmread(str(dcm_path), force=True)
                uid = getattr(ds, 'SeriesInstanceUID', 'UNKNOWN')
                if uid not in series_groups:
                    series_groups[uid] = []
                series_groups[uid].append((dcm_path, ds))
            except Exception as e:
                print(f"⚠️ 读取DICOM失败: {dcm_path}, 错误: {e}")
                continue

        if not series_groups:
            return None

        # 选择切片数量最多的序列
        best_series = max(series_groups.values(), key=lambda x: len(x))
        print(f"✅ 选择序列: {len(best_series)} 张切片")

        # 提取并排序
        slices = []
        for dcm_path, ds in best_series:
            arr = ds.pixel_array.astype(np.float32)
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                arr = arr * ds.RescaleSlope + ds.RescaleIntercept

            # 获取Z轴位置（优先使用 ImagePositionPatient）
            if hasattr(ds, 'ImagePositionPatient'):
                z_pos = float(ds.ImagePositionPatient[2])
            elif hasattr(ds, 'InstanceNumber'):
                z_pos = int(ds.InstanceNumber)
            else:
                z_pos = 0
            slices.append((z_pos, arr))

        # 按Z轴位置排序
        slices.sort(key=lambda x: x[0])
        z_positions = [s[0] for s in slices]
        if len(z_positions) > 1:
            intervals = np.diff(z_positions)
            print(f"Z轴间隔（前10个）: {intervals[:10]}")
            print(f"Z轴间隔标准差: {np.std(intervals):.2f}")

        volume = np.stack([s[1] for s in slices], axis=0)  # (D, H, W)

        # 归一化（用于显示）
        volume = np.clip(volume, -100, 100)
        vmin, vmax = volume.min(), volume.max()
        if vmax - vmin > 0:
            volume = (volume - vmin) / (vmax - vmin)

        print(f"✅ 体积形状: {volume.shape}")
        return volume

def extract_center_slices(volume):
    """提取三视图中心切片，并旋转使方向对齐"""
    d, h, w = volume.shape
    axial = volume[d//2, :, :]
    coronal = volume[:, h//2, :]
    sagittal = volume[:, :, w//2]
    # 旋转（与API保持一致）
    coronal = np.rot90(coronal, k=1)
    sagittal = np.rot90(sagittal, k=-1)
    return axial, coronal, sagittal

def create_probability_bar(predictions):
    """创建概率条形图（Plotly）"""
    labels = [p['label'] for p in predictions]
    probs = [p['probability'] for p in predictions]
    colors = ['#2E86AB' if p['positive'] else '#A0A0A0' for p in predictions]
    fig = go.Figure(data=[go.Bar(x=labels, y=probs, marker_color=colors)])
    fig.update_layout(
        title='病灶概率',
        yaxis_title='概率',
        yaxis_range=[0, 1],
        height=300,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig

# ==================== 核心处理函数 ====================
def process_zip_file(file):
    if file is None:
        return None, None, "未上传文件", None

    zip_path = file.name if hasattr(file, 'name') else file

    # 1. 显示三视图
    volume = load_dicom_from_zip(zip_path)
    if volume is None:
        return None, None, "无法读取DICOM文件", None
    axial, coronal, sagittal = extract_center_slices(volume)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(axial, cmap='gray', interpolation='nearest')
    axes[0].set_title('轴位 (Axial)')
    axes[0].axis('off')
    axes[1].imshow(coronal, cmap='gray', interpolation='nearest')
    axes[1].set_title('冠状 (Coronal)')
    axes[1].axis('off')
    axes[2].imshow(sagittal, cmap='gray', interpolation='nearest')
    axes[2].set_title('矢状 (Sagittal)')
    axes[2].axis('off')
    plt.tight_layout()
    img_path = os.path.join(TEMP_DIR, "mpr_views.png")
    plt.savefig(img_path, dpi=100, bbox_inches='tight')
    plt.close()

    # 2. 调用API进行预测
    url = f"{API_URL}/predict/volume"
    with open(zip_path, 'rb') as f:
        files = {'file': ('volume.zip', f, 'application/zip')}
        try:
            response = requests.post(url, files=files, timeout=30)
            if response.status_code == 200:
                result = response.json()
                predictions = result['predictions']
                prob_fig = create_probability_bar(predictions)
                positive = [p['label'] for p in predictions if p['positive']]
                if positive:
                    report_text = f"检出阳性病灶：{', '.join(positive)}\n\n"
                    report_text += "详细概率如下：\n"
                    for p in predictions:
                        report_text += f"{p['label']}: {p['probability']:.3f}\n"
                else:
                    report_text = "未检出明显病灶。"
                # 返回4个值（与输出组件匹配）
                return img_path, prob_fig, report_text, f"耗时 {result['elapsed_ms']:.0f} ms"
            else:
                return None, None, f"API调用失败: {response.text}", None
        except Exception as e:
            return None, None, f"错误: {str(e)}", None

def process_dicom_file(file):
    if file is None:
        return None, None, "未上传文件", None

    dcm_path = file.name if hasattr(file, 'name') else file

    # 显示DICOM图像
    ds = pydicom.dcmread(dcm_path)
    img = ds.pixel_array.astype(np.float32)
    img = np.clip(img, -100, 100)
    vmin, vmax = img.min(), img.max()
    if vmax - vmin > 0:
        img = (img - vmin) / (vmax - vmin)
    img_path = os.path.join(TEMP_DIR, "dicom_slice.png")
    plt.imsave(img_path, img, cmap='gray')

    # 调用 /predict/dicom
    url = f"{API_URL}/predict/dicom"
    with open(dcm_path, 'rb') as f:
        files = {'file': ('slice.dcm', f, 'application/octet-stream')}
        try:
            response = requests.post(url, files=files, timeout=30)
            if response.status_code == 200:
                result = response.json()
                predictions = result['predictions']
                prob_fig = create_probability_bar(predictions)
                # 生成报告
                report_url = f"{API_URL}/predict/report"
                with open(dcm_path, 'rb') as f2:
                    report_files = {'file': ('slice.dcm', f2, 'application/octet-stream')}
                    data = {'case_id': 'DEMO', 'patient_name': 'Demo', 'patient_age': '40'}
                    report_resp = requests.post(report_url, files=report_files, data=data, timeout=30)
                    report_text = ""
                    if report_resp.status_code == 200:
                        report_text = report_resp.json().get('report', '')
                    else:
                        report_text = f"报告生成失败: {report_resp.text}"
                return img_path, prob_fig, report_text, f"耗时 {result['elapsed_ms']:.0f} ms"
            else:
                return None, None, f"API调用失败: {response.text}", None
        except Exception as e:
            return None, None, f"错误: {str(e)}", None

# ==================== Gradio 界面 ====================
with gr.Blocks(title="BrainCT 脑部CT分析系统") as demo:
    gr.Markdown("# 🧠 BrainCT 脑部CT病灶识别与报告系统")
    gr.Markdown("上传DICOM序列（ZIP）或单个DICOM文件，系统将进行多标签分类并生成诊断报告。")

    with gr.Tab("体积分析（三视图MPR）"):
        with gr.Row():
            with gr.Column(scale=1):
                zip_input = gr.File(label="上传ZIP文件（包含所有DICOM切片）", file_types=[".zip"])
                zip_btn = gr.Button("开始分析", variant="primary")
            with gr.Column(scale=1):
                zip_output_img = gr.Image(label="三视图 (轴位/冠状/矢状)", type="filepath")
                zip_output_prob = gr.Plot(label="病灶概率")
                zip_output_report = gr.Textbox(label="诊断报告", lines=10)
                zip_output_time = gr.Textbox(label="耗时")
        zip_btn.click(
            process_zip_file,
            inputs=zip_input,
            outputs=[zip_output_img, zip_output_prob, zip_output_report, zip_output_time]
        )

    with gr.Tab("单张切片分析"):
        with gr.Row():
            with gr.Column(scale=1):
                dcm_input = gr.File(label="上传单个DICOM文件 (.dcm)", file_types=[".dcm"])
                dcm_btn = gr.Button("开始分析", variant="primary")
            with gr.Column(scale=1):
                dcm_output_img = gr.Image(label="CT切片", type="filepath")
                dcm_output_prob = gr.Plot(label="病灶概率")
                dcm_output_report = gr.Textbox(label="诊断报告", lines=10)
                dcm_output_time = gr.Textbox(label="耗时")
        dcm_btn.click(
            process_dicom_file,
            inputs=dcm_input,
            outputs=[dcm_output_img, dcm_output_prob, dcm_output_report, dcm_output_time]
        )

    gr.Markdown("---")
    gr.Markdown("**说明**：该系统基于深度学习模型，检测9类脑部病灶并生成结构化报告。仅供研究参考，不作为临床诊断依据。")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)