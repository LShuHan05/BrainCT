import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
import requests
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import io
import zipfile
from pathlib import Path
import tempfile
import plotly.graph_objects as go
from PIL import Image

# FastAPI 服务地址
API_URL = "http://127.0.0.1:8000"

# ==================== 辅助函数 ====================
def load_dicom_from_zip(zip_bytes):
    """从ZIP字节流中提取所有DICOM，返回体积数组（用于显示）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "volume.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_bytes)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        dcm_files = sorted(Path(tmpdir).glob("*.dcm"))
        if not dcm_files:
            return None
        slices = []
        for dcm_path in dcm_files:
            ds = pydicom.dcmread(str(dcm_path))
            arr = ds.pixel_array.astype(np.float32)
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                arr = arr * ds.RescaleSlope + ds.RescaleIntercept
            slices.append(arr)
        volume = np.stack(slices, axis=0)
        # 归一化用于显示
        volume = np.clip(volume, -100, 100)
        vmin, vmax = volume.min(), volume.max()
        if vmax - vmin > 0:
            volume = (volume - vmin) / (vmax - vmin)
        return volume

def extract_center_slices(volume):
    """提取三视图中心切片用于显示"""
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

def display_report_text(report):
    """格式化报告显示"""
    return report

# ==================== 核心处理函数 ====================
def process_zip_file(file):
    """
    处理上传的ZIP文件：显示三视图 + 调用API预测 + 显示报告
    """
    if file is None:
        return None, None, None, None, None, None

    # 读取ZIP字节
    zip_bytes = file

    # 1. 显示三视图
    volume = load_dicom_from_zip(zip_bytes)
    if volume is None:
        return "无法读取DICOM文件", None, None, None, None
    axial, coronal, sagittal = extract_center_slices(volume)

    # 创建三视图图像（matplotlib）
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(axial, cmap='gray')
    axes[0].set_title('轴位 (Axial)')
    axes[0].axis('off')
    axes[1].imshow(coronal, cmap='gray')
    axes[1].set_title('冠状 (Coronal)')
    axes[1].axis('off')
    axes[2].imshow(sagittal, cmap='gray')
    axes[2].set_title('矢状 (Sagittal)')
    axes[2].axis('off')
    plt.tight_layout()
    # 保存为临时图片
    img_path = "/tmp/mpr_views.png"
    plt.savefig(img_path, dpi=100, bbox_inches='tight')
    plt.close()

    # 2. 调用API进行预测（上传ZIP）
    url = f"{API_URL}/predict/volume"
    files = {'file': ('volume.zip', zip_bytes, 'application/zip')}
    try:
        response = requests.post(url, files=files)
        if response.status_code == 200:
            result = response.json()
            predictions = result['predictions']
            # 生成概率条形图
            prob_fig = create_probability_bar(predictions)
            # 生成报告（调用 /predict/report，但这里需要先有预测结果）
            # 简便起见，我们单独再调用 report 端点（使用第一个DICOM文件）
            # 但为了演示，此处我们可以直接显示预测结果
            report_text = "（报告生成需要调用 /predict/report 端点，由于需要单个DICOM文件，此处暂略）"
            # 也可以从 predictions 生成简单的文本报告
            positive = [p['label'] for p in predictions if p['positive']]
            if positive:
                report_text = f"检出阳性病灶：{', '.join(positive)}\n\n"
                report_text += "详细概率如下：\n"
                for p in predictions:
                    report_text += f"{p['label']}: {p['probability']:.3f}\n"
            else:
                report_text = "未检出明显病灶。"
            return img_path, prob_fig, report_text, f"耗时 {result['elapsed_ms']:.0f} ms", None
        else:
            return None, None, f"API调用失败: {response.text}", None, None
    except Exception as e:
        return None, None, f"错误: {str(e)}", None, None

def process_dicom_file(file):
    """
    处理单个DICOM文件：显示图像 + 调用 /predict/dicom + 报告
    """
    if file is None:
        return None, None, None, None

    dcm_bytes = file
    # 显示DICOM图像（使用pydicom读取）
    ds = pydicom.dcmread(io.BytesIO(dcm_bytes))
    img = ds.pixel_array.astype(np.float32)
    # 归一化显示
    img = np.clip(img, -100, 100)
    vmin, vmax = img.min(), img.max()
    if vmax - vmin > 0:
        img = (img - vmin) / (vmax - vmin)
    # 保存为临时图像
    img_path = "/tmp/dicom_slice.png"
    plt.imsave(img_path, img, cmap='gray')

    # 调用 /predict/dicom
    url = f"{API_URL}/predict/dicom"
    files = {'file': ('slice.dcm', dcm_bytes, 'application/octet-stream')}
    try:
        response = requests.post(url, files=files)
        if response.status_code == 200:
            result = response.json()
            predictions = result['predictions']
            prob_fig = create_probability_bar(predictions)
            # 生成报告（调用 /predict/report）
            report_url = f"{API_URL}/predict/report"
            report_files = {'file': ('slice.dcm', dcm_bytes, 'application/octet-stream')}
            data = {'case_id': 'DEMO', 'patient_name': 'Demo', 'patient_age': '40'}
            report_resp = requests.post(report_url, files=report_files, data=data)
            report_text = ""
            if report_resp.status_code == 200:
                report_text = report_resp.json().get('report', '')
            else:
                report_text = "报告生成失败"
            return img_path, prob_fig, report_text, f"耗时 {result['elapsed_ms']:.0f} ms"
        else:
            return None, None, f"API调用失败: {response.text}", None
    except Exception as e:
        return None, None, f"错误: {str(e)}", None

# ==================== Gradio 界面 ====================
with gr.Blocks(title="BrainCT 脑部CT分析系统", theme=gr.themes.Soft()) as demo:
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

# 启动服务
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)