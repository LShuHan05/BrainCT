"""
脑部病灶分类推理 API（增强版）
功能：
- 健康检查、模型信息
- 单个 DICOM/图像预测
- 批处理预测
- 综合推理+报告生成
- 三视图（MPR）体积预测（新增）
- 日志记录、异常处理、模型版本控制
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
from pydantic import BaseModel, Field
from PIL import Image
import io

# 导入项目模块
from Conf.Config import *
from Model.LesionClassifier import LesionClassifier2D
from Services.MedicalReportGenerator import MedicalReportGenerator
from Services.LLMService import LLMService
from Utils.InferenceLogger import InferenceLogger

# ==================== 初始化 ====================
app = FastAPI(title="BrainCT 病灶分类 API", version="2.0.0")
logger = InferenceLogger(API_LOG_DIR)

# 全局变量
models = {}
model_metadata = {}
current_version = DEFAULT_MODEL_VERSION

# 初始化 LLM 服务（显式传入 API Key）
llm_service = LLMService(
    api_key=os.getenv("DEEPSEEK_API_KEY", "sk-gkhdynknixxmgjuxbjywzdmpacinebnushnlyjvjdstviend"),
    model=DEEPSEEK_MODEL
)
report_gen = MedicalReportGenerator(llm_service=llm_service)

# ==================== 模型加载 ====================
def load_model(version: str):
    """加载指定版本的模型"""
    weight_path = MODEL_VERSIONS.get(version)
    if not weight_path or not os.path.exists(weight_path):
        raise FileNotFoundError(f"Model weight not found: {weight_path}")

    model = LesionClassifier2D(
        num_classes=NUM_CLASSES,
        input_channels=INPUT_CHANNELS,
        use_three_views=USE_THREE_VIEWS
    )
    checkpoint = torch.load(weight_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    return model

@app.on_event("startup")
async def startup_event():
    """启动时加载默认模型"""
    global models, model_metadata
    for ver, path in MODEL_VERSIONS.items():
        try:
            models[ver] = load_model(ver)
            # 读取元数据
            if os.path.exists(path):
                ckpt = torch.load(path, map_location='cpu')
                model_metadata[ver] = {
                    'best_f1': ckpt.get('best_f1', 0.0),
                    'epoch': ckpt.get('epoch', 0),
                    'version': ver
                }
        except Exception as e:
            print(f"⚠️  Failed to load model {ver}: {e}")

# ==================== 数据预处理 ====================
def preprocess_dicom_bytes(dcm_data: bytes) -> torch.Tensor:
    """处理单个 DICOM 文件字节流"""
    ds = pydicom.dcmread(io.BytesIO(dcm_data))
    pixel_array = ds.pixel_array.astype(np.float32)
    if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
        pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept
    pixel_array = np.clip(pixel_array, -100, 100)
    vmin, vmax = pixel_array.min(), pixel_array.max()
    if vmax - vmin > 0:
        pixel_array = (pixel_array - vmin) / (vmax - vmin)
    h, w = pixel_array.shape
    target_h, target_w = TARGET_SIZE_2D
    pixel_array = zoom(pixel_array, (target_h/h, target_w/w), order=1)
    tensor = torch.from_numpy(pixel_array).unsqueeze(0).unsqueeze(0).float()
    if INPUT_CHANNELS == 3:
        tensor = tensor.repeat(1, 3, 1, 1)
    return tensor.to(DEVICE)

def preprocess_image_bytes(image_data: bytes) -> torch.Tensor:
    """处理通用图像（PNG/JPG）"""
    img = Image.open(io.BytesIO(image_data)).convert('L')
    pixel_array = np.array(img, dtype=np.float32) / 255.0
    h, w = pixel_array.shape
    target_h, target_w = TARGET_SIZE_2D
    pixel_array = zoom(pixel_array, (target_h/h, target_w/w), order=1)
    tensor = torch.from_numpy(pixel_array).unsqueeze(0).unsqueeze(0).float()
    if INPUT_CHANNELS == 3:
        tensor = tensor.repeat(1, 3, 1, 1)
    return tensor.to(DEVICE)

def preprocess_volume_for_mpr(volume: np.ndarray) -> np.ndarray:
    """
    对 3D 体积进行标准化预处理
    :param volume: (D, H, W) 原始 HU 值数组
    :return: 归一化后的体积 (D, H, W) float32
    """
    volume = np.clip(volume, -100, 100).astype(np.float32)
    vmin, vmax = volume.min(), volume.max()
    if vmax - vmin > 0:
        volume = (volume - vmin) / (vmax - vmin)
    return volume

def extract_mpr_views(volume: np.ndarray, target_size=(256, 256)):
    """
    从 3D 体积中提取轴位、冠状、矢状中心切片，并堆叠为 3 通道图像
    :param volume: (D, H, W) 归一化后的体积
    :param target_size: 输出图像尺寸 (H, W)
    :return: (3, H, W) numpy 数组
    """
    d, h, w = volume.shape
    # 中心切片
    axial = volume[d//2, :, :]          # (H, W)
    coronal = volume[:, h//2, :]        # (D, W)
    sagittal = volume[:, :, w//2]       # (D, H)

    # 旋转使冠状和矢状与轴位解剖方向一致（经合成数据验证）
    coronal = np.rot90(coronal, k=1)    # 顺时针旋转90度
    sagittal = np.rot90(sagittal, k=-1) # 逆时针旋转90度

    # 统一 resize 到目标尺寸
    def resize_view(view):
        h0, w0 = view.shape
        return zoom(view, (target_size[0]/h0, target_size[1]/w0), order=1)

    axial = resize_view(axial)
    coronal = resize_view(coronal)
    sagittal = resize_view(sagittal)

    # 堆叠为 3 通道
    three_view = np.stack([axial, coronal, sagittal], axis=0)  # (3, H, W)
    return three_view.astype(np.float32)

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

# ==================== 端点 ====================
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": len(models) > 0, "device": DEVICE}

@app.get("/models")
async def list_models():
    return {"versions": list(models.keys()), "current": current_version, "metadata": model_metadata}

@app.post("/predict/dicom")
async def predict_dicom(
    file: UploadFile = File(...),
    version: Optional[str] = None
):
    start_time = time.time()
    ver = version or current_version
    if ver not in models:
        raise HTTPException(status_code=400, detail=f"Model version {ver} not loaded")

    try:
        dcm_data = await file.read()
        tensor = preprocess_dicom_bytes(dcm_data)
        with torch.no_grad():
            logits = models[ver](tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        predictions = []
        for label, prob in zip(LESION_LABELS, probs):
            predictions.append({
                "label": label,
                "probability": float(prob),
                "positive": bool(prob >= POSITIVE_THRESHOLD)
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
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(
    files: List[UploadFile] = File(...),
    version: Optional[str] = None
):
    start_time = time.time()
    ver = version or current_version
    if ver not in models:
        raise HTTPException(status_code=400, detail=f"Model version {ver} not loaded")

    results = []
    for file in files:
        try:
            dcm_data = await file.read()
            tensor = preprocess_dicom_bytes(dcm_data)
            with torch.no_grad():
                logits = models[ver](tensor)
                probs = torch.sigmoid(logits).cpu().numpy()[0]
            preds = []
            for label, prob in zip(LESION_LABELS, probs):
                preds.append({
                    "label": label,
                    "probability": float(prob),
                    "positive": bool(prob >= POSITIVE_THRESHOLD)
                })
            results.append({"filename": file.filename, "predictions": preds})
        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})

    elapsed = (time.time() - start_time) * 1000
    logger.log({
        "endpoint": "/predict/batch",
        "count": len(files),
        "elapsed_ms": elapsed
    })
    return BatchPredictionResponse(
        success=True,
        results=results,
        total=len(files),
        elapsed_ms=elapsed
    )

@app.post("/predict/volume")
async def predict_volume(
    file: UploadFile = File(...),  # 上传一个 ZIP 文件
    version: Optional[str] = None
):
    """
    上传一个包含完整 DICOM 序列的 ZIP 文件，进行三视图（MPR）推理
    """
    start_time = time.time()
    ver = version or current_version
    if ver not in models:
        raise HTTPException(status_code=400, detail=f"Model version {ver} not loaded")

    # 创建临时目录解压 ZIP
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "volume.zip")
        with open(zip_path, "wb") as f:
            f.write(await file.read())

        # 解压
        extract_dir = os.path.join(tmpdir, "dicom_series")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        # 读取所有 DICOM 文件（按文件名排序，假设顺序对应空间位置）
        dcm_files = sorted(Path(extract_dir).glob("*.dcm"))
        if not dcm_files:
            raise HTTPException(status_code=400, detail="No DICOM files found in ZIP")

        # 加载所有切片，构建 3D 体积
        slices = []
        for dcm_path in dcm_files:
            ds = pydicom.dcmread(str(dcm_path))
            arr = ds.pixel_array.astype(np.float32)
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                arr = arr * ds.RescaleSlope + ds.RescaleIntercept
            slices.append(arr)
        volume = np.stack(slices, axis=0)  # (D, H, W)

        # 预处理体积
        volume = preprocess_volume_for_mpr(volume)

        # 提取三视图
        three_view = extract_mpr_views(volume, target_size=TARGET_SIZE_2D)

        # 转换为张量 (1, 3, H, W)
        tensor = torch.from_numpy(three_view).unsqueeze(0).float().to(DEVICE)

        # 推理
        with torch.no_grad():
            logits = models[ver](tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        predictions = []
        for label, prob in zip(LESION_LABELS, probs):
            predictions.append({
                "label": label,
                "probability": float(prob),
                "positive": bool(prob >= POSITIVE_THRESHOLD)
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

@app.post("/predict/report")
async def predict_with_report(
    file: UploadFile = File(...),
    case_id: str = "Unknown",
    patient_name: Optional[str] = None,
    patient_age: Optional[int] = None,
    version: Optional[str] = None,
    background_tasks: BackgroundTasks = None
):
    """
    一站式：推理 + 生成医学报告（支持单张 DICOM）
    """
    start_time = time.time()
    ver = version or current_version
    if ver not in models:
        raise HTTPException(status_code=400, detail=f"Model version {ver} not loaded")

    try:
        dcm_data = await file.read()
        tensor = preprocess_dicom_bytes(dcm_data)
        with torch.no_grad():
            logits = models[ver](tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        predictions = {}
        for label, prob in zip(LESION_LABELS, probs):
            predictions[label] = float(prob)

        # 生成报告
        patient_info = {}
        if patient_name:
            patient_info['name'] = patient_name
        if patient_age:
            patient_info['age'] = patient_age

        # 调用同步的报告生成（已修改为同步）
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
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")