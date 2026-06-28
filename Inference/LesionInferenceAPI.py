# Inference/LesionInferenceAPI.py
"""
脑部病灶分类推理 API（完整版）
- 加载最终模型 final_best_lesion.pth 及对应阈值
- 支持单张 DICOM、批量、三视图体积、报告生成
- 兼容 Gradio 前端调用
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import zipfile
import tempfile
from pathlib import Path
from typing import Optional, List
import numpy as np
import torch
import pydicom
from scipy.ndimage import zoom
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import io

# ---------- 项目内部导入 ----------
from Conf.Config import *
from Model.LesionClassifier import LesionClassifier2D
from Services.MedicalReportGenerator import MedicalReportGenerator
from Services.LLMService import LLMService
from Utils.InferenceLogger import InferenceLogger

# ==================== 初始化 ====================
app = FastAPI(
    title="BrainCT 病灶分类 API (Final Model)",
    version="3.0.0",
    description="基于 2D MIL + ResNet50 的脑部CT多标签分类服务"
)
logger = InferenceLogger(API_LOG_DIR)

# ---------- 全局变量 ----------
models = {}                 # 版本 -> 模型实例
model_metadata = {}         # 版本 -> 元数据（F1, epoch, thresholds）
model_thresholds = {}       # 版本 -> 类别阈值列表
current_version = DEFAULT_MODEL_VERSION

# ---------- LLM 服务 ----------
llm_service = LLMService(
    api_key=os.getenv("DEEPSEEK_API_KEY", "sk-gkhdynknixxmgjuxbjywzdmpacinebnushnlyjvjdstviend"),
    model=DEEPSEEK_MODEL
)
report_gen = MedicalReportGenerator(llm_service=llm_service)

# ==================== 模型加载 ====================
def load_model(version: str):
    """加载指定版本的模型及对应的最佳阈值"""
    weight_path = MODEL_VERSIONS.get(version)
    if not weight_path or not os.path.exists(weight_path):
        raise FileNotFoundError(f"模型权重不存在: {weight_path}")

    # 加载完整 checkpoint
    checkpoint = torch.load(weight_path, map_location=DEVICE, weights_only=False)

    # 构建模型结构（与训练时一致）
    model = LesionClassifier2D(
        num_classes=NUM_CLASSES,
        input_channels=INPUT_CHANNELS,
        use_three_views=USE_THREE_VIEWS
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()

    # 提取阈值（若不存在则使用默认 0.5）
    thresholds = checkpoint.get('best_thresholds', [POSITIVE_THRESHOLD] * NUM_CLASSES)
    best_f1 = checkpoint.get('best_f1', 0.0)

    # 保存元数据
    model_metadata[version] = {
        'best_f1': best_f1,
        'epoch': checkpoint.get('epoch', 0),
        'version': version,
        'thresholds': thresholds
    }
    model_thresholds[version] = thresholds

    print(f"✅ 模型加载成功: {version}")
    print(f"   Best F1: {best_f1:.4f}")
    print(f"   Thresholds: {thresholds}")
    return model

@app.on_event("startup")
async def startup_event():
    """应用启动时加载默认模型"""
    global models
    for ver in MODEL_VERSIONS:
        try:
            models[ver] = load_model(ver)
        except Exception as e:
            print(f"⚠️ 模型 {ver} 加载失败: {e}")

# ==================== 数据预处理函数 ====================
def preprocess_dicom_bytes(dcm_data: bytes) -> torch.Tensor:
    """从字节流读取 DICOM，预处理为模型输入张量"""
    ds = pydicom.dcmread(io.BytesIO(dcm_data))
    pixel_array = ds.pixel_array.astype(np.float32)

    # 转换为 HU
    if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
        pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept

    # 裁剪至脑组织范围并归一化
    pixel_array = np.clip(pixel_array, -100, 100)
    vmin, vmax = pixel_array.min(), pixel_array.max()
    if vmax - vmin > 0:
        pixel_array = (pixel_array - vmin) / (vmax - vmin)

    # 缩放到目标尺寸
    h, w = pixel_array.shape
    target_h, target_w = TARGET_SIZE_2D
    pixel_array = zoom(pixel_array, (target_h / h, target_w / w), order=1)

    # 转为张量，添加 batch 和 channel 维度
    tensor = torch.from_numpy(pixel_array).unsqueeze(0).unsqueeze(0).float()

    # 若模型需要三通道输入，复制通道
    if INPUT_CHANNELS == 3:
        tensor = tensor.repeat(1, 3, 1, 1)

    return tensor.to(DEVICE)

def preprocess_image_bytes(image_data: bytes) -> torch.Tensor:
    """处理 PNG/JPG 图像（灰度）"""
    img = Image.open(io.BytesIO(image_data)).convert('L')
    pixel_array = np.array(img, dtype=np.float32) / 255.0
    h, w = pixel_array.shape
    target_h, target_w = TARGET_SIZE_2D
    pixel_array = zoom(pixel_array, (target_h / h, target_w / w), order=1)
    tensor = torch.from_numpy(pixel_array).unsqueeze(0).unsqueeze(0).float()
    if INPUT_CHANNELS == 3:
        tensor = tensor.repeat(1, 3, 1, 1)
    return tensor.to(DEVICE)

def preprocess_volume_for_mpr(volume: np.ndarray) -> np.ndarray:
    """标准化 3D 体积（HU 裁剪 + 归一化）"""
    volume = np.clip(volume, -100, 100).astype(np.float32)
    vmin, vmax = volume.min(), volume.max()
    if vmax - vmin > 0:
        volume = (volume - vmin) / (vmax - vmin)
    return volume

def extract_mpr_views(volume: np.ndarray, target_size=(256, 256)) -> np.ndarray:
    """从 3D 体积提取轴位、冠状、矢状三视图，堆叠为 3 通道"""
    d, h, w = volume.shape
    axial = volume[d // 2, :, :]
    coronal = volume[:, h // 2, :]
    sagittal = volume[:, :, w // 2]

    # 旋转使方向对齐（与训练时一致）
    coronal = np.rot90(coronal, k=1)
    sagittal = np.rot90(sagittal, k=-1)

    def resize_view(view):
        h0, w0 = view.shape
        return zoom(view, (target_size[0] / h0, target_size[1] / w0), order=1)

    axial = resize_view(axial)
    coronal = resize_view(coronal)
    sagittal = resize_view(sagittal)

    return np.stack([axial, coronal, sagittal], axis=0).astype(np.float32)

# ==================== 响应模型 ====================
class PredictionResponse(BaseModel):
    success: bool
    predictions: List[dict]
    model_version: str
    elapsed_ms: float
    message: str = ""

class BatchPredictionResponse(BaseModel):
    success: bool
    results: List[dict]
    total: int
    elapsed_ms: float

class ReportResponse(BaseModel):
    success: bool
    case_id: str
    report: str
    model_version: str
    elapsed_ms: float

# ==================== API 端点 ====================
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": len(models) > 0, "device": DEVICE}

@app.get("/models")
async def list_models():
    return {"versions": list(models.keys()), "current": current_version, "metadata": model_metadata}

# ---------- 单张 DICOM 预测 ----------
@app.post("/predict/dicom")
async def predict_dicom(
    file: UploadFile = File(...),
    version: Optional[str] = None
):
    start_time = time.time()
    ver = version or current_version
    if ver not in models:
        raise HTTPException(status_code=400, detail=f"模型版本 {ver} 未加载")

    thresholds = model_thresholds.get(ver, [POSITIVE_THRESHOLD] * NUM_CLASSES)

    try:
        dcm_data = await file.read()
        tensor = preprocess_dicom_bytes(dcm_data)
        with torch.no_grad():
            logits = models[ver](tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        predictions = []
        for i, (label, prob) in enumerate(zip(LESION_LABELS, probs)):
            predictions.append({
                "label": label,
                "probability": float(prob),
                "positive": bool(prob >= thresholds[i])
            })

        elapsed = (time.time() - start_time) * 1000
        logger.log({
            "endpoint": "/predict/dicom",
            "filename": file.filename,
            "version": ver,
            "elapsed_ms": elapsed,
            "positive_count": sum(1 for p in predictions if p['positive'])
        })

        return PredictionResponse(
            success=True,
            predictions=predictions,
            model_version=ver,
            elapsed_ms=elapsed
        )
    except Exception as e:
        logger.log({"endpoint": "/predict/dicom", "filename": file.filename, "error": str(e)})
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")

# ---------- 批量 DICOM 预测 ----------
@app.post("/predict/batch")
async def predict_batch(
    files: List[UploadFile] = File(...),
    version: Optional[str] = None
):
    start_time = time.time()
    ver = version or current_version
    if ver not in models:
        raise HTTPException(status_code=400, detail=f"模型版本 {ver} 未加载")

    thresholds = model_thresholds.get(ver, [POSITIVE_THRESHOLD] * NUM_CLASSES)
    results = []

    for file in files:
        try:
            dcm_data = await file.read()
            tensor = preprocess_dicom_bytes(dcm_data)
            with torch.no_grad():
                logits = models[ver](tensor)
                probs = torch.sigmoid(logits).cpu().numpy()[0]
            preds = []
            for i, (label, prob) in enumerate(zip(LESION_LABELS, probs)):
                preds.append({
                    "label": label,
                    "probability": float(prob),
                    "positive": bool(prob >= thresholds[i])
                })
            results.append({"filename": file.filename, "predictions": preds})
        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})

    elapsed = (time.time() - start_time) * 1000
    logger.log({"endpoint": "/predict/batch", "count": len(files), "elapsed_ms": elapsed})
    return BatchPredictionResponse(
        success=True,
        results=results,
        total=len(files),
        elapsed_ms=elapsed
    )

# ---------- 三视图体积预测（ZIP 上传） ----------
@app.post("/predict/volume")
async def predict_volume(
    file: UploadFile = File(...),
    version: Optional[str] = None
):
    start_time = time.time()
    ver = version or current_version
    if ver not in models:
        raise HTTPException(status_code=400, detail=f"模型版本 {ver} 未加载")

    thresholds = model_thresholds.get(ver, [POSITIVE_THRESHOLD] * NUM_CLASSES)

    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "volume.zip")
        with open(zip_path, "wb") as f:
            f.write(await file.read())

        extract_dir = os.path.join(tmpdir, "dicom_series")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        dcm_files = sorted(Path(extract_dir).glob("*.dcm"))
        if not dcm_files:
            raise HTTPException(status_code=400, detail="ZIP 中未找到 DICOM 文件")

        slices = []
        for dcm_path in dcm_files:
            ds = pydicom.dcmread(str(dcm_path))
            arr = ds.pixel_array.astype(np.float32)
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                arr = arr * ds.RescaleSlope + ds.RescaleIntercept
            slices.append(arr)
        volume = np.stack(slices, axis=0)  # (D, H, W)

        volume = preprocess_volume_for_mpr(volume)
        three_view = extract_mpr_views(volume, target_size=TARGET_SIZE_2D)
        tensor = torch.from_numpy(three_view).unsqueeze(0).float().to(DEVICE)

        with torch.no_grad():
            logits = models[ver](tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        predictions = []
        for i, (label, prob) in enumerate(zip(LESION_LABELS, probs)):
            predictions.append({
                "label": label,
                "probability": float(prob),
                "positive": bool(prob >= thresholds[i])
            })

        elapsed = (time.time() - start_time) * 1000
        logger.log({
            "endpoint": "/predict/volume",
            "filename": file.filename,
            "version": ver,
            "elapsed_ms": elapsed,
            "slices_count": len(dcm_files)
        })

        return PredictionResponse(
            success=True,
            predictions=predictions,
            model_version=ver,
            elapsed_ms=elapsed
        )

# ---------- 预测 + 报告生成 ----------
@app.post("/predict/report")
async def predict_with_report(
    file: UploadFile = File(...),
    case_id: str = "Unknown",
    patient_name: Optional[str] = None,
    patient_age: Optional[int] = None,
    version: Optional[str] = None
):
    start_time = time.time()
    ver = version or current_version
    if ver not in models:
        raise HTTPException(status_code=400, detail=f"模型版本 {ver} 未加载")

    thresholds = model_thresholds.get(ver, [POSITIVE_THRESHOLD] * NUM_CLASSES)

    try:
        dcm_data = await file.read()
        tensor = preprocess_dicom_bytes(dcm_data)
        with torch.no_grad():
            logits = models[ver](tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        # 构建预测字典（用于报告）
        predictions = {}
        for i, (label, prob) in enumerate(zip(LESION_LABELS, probs)):
            predictions[label] = float(prob)

        # 生成报告（同步调用）
        patient_info = {}
        if patient_name:
            patient_info['name'] = patient_name
        if patient_age:
            patient_info['age'] = patient_age

        report_text = report_gen.generate_report(
            case_id=case_id,
            predictions=predictions,
            patient_info=patient_info
        )

        elapsed = (time.time() - start_time) * 1000
        logger.log({
            "endpoint": "/predict/report",
            "case_id": case_id,
            "version": ver,
            "elapsed_ms": elapsed
        })

        return ReportResponse(
            success=True,
            case_id=case_id,
            report=report_text,
            model_version=ver,
            elapsed_ms=elapsed
        )
    except Exception as e:
        logger.log({"endpoint": "/predict/report", "case_id": case_id, "error": str(e)})
        raise HTTPException(status_code=500, detail=f"报告生成失败: {str(e)}")