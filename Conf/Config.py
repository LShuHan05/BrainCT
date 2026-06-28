# Conf/Config.py
import os
import torch

# ====================== 路径配置 ======================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SAVE_DIR = os.path.join(BASE_DIR, "run")
WEIGHT_SAVE_DIR = os.path.join(SAVE_DIR, "weights")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(WEIGHT_SAVE_DIR, exist_ok=True)
WEIGHT_FILE = "best.pth"

CT_PATH = os.path.join(BASE_DIR, "datasets", "CT")
MASK_PATH = os.path.join(BASE_DIR, "datasets", "MASK")

# ====================== 完整数据集路径 ======================
CQ500_ORIG_ROOT = os.path.join(BASE_DIR, "CQ500_orig")
PREPROCESS_OUTPUT_DIR = os.path.join(BASE_DIR, "datasets_processed")
NIFTI_DIR = os.path.join(PREPROCESS_OUTPUT_DIR, "nifti")
METADATA_DIR = os.path.join(PREPROCESS_OUTPUT_DIR, "metadata")
METADATA_CSV = os.path.join(METADATA_DIR, "dataset_metadata.csv")

CQ500_DATA_ROOT = os.path.join(BASE_DIR, "datasets_filtered")
CQ500_ANNOTATION_FILE = os.path.join(BASE_DIR, "md", "reads.csv")
ANNOTATION_FILE = CQ500_ANNOTATION_FILE

# ====================== 任务模式选择 ======================
TASK_MODE = "lesion_classification"

# ====================== 训练超参数 ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 12
EPOCHS = 100
LEARNING_RATE = 2e-4
VAL_SPLIT = 0.2
LR = 2e-4

# ====================== 优化配置 ======================
USE_AUXILIARY = True
USE_FOCAL_LOSS = True
GRAD_CLIP = 1.0
EARLY_STOP_PATIENCE = 25

# ====================== 3D模型配置（保留兼容） ======================
TARGET_SHAPE_3D = (128, 128, 64)
EPOCHS_3D = 120

# ====================== 损失函数配置 ======================
FOCAL_ALPHA = 0.75
FOCAL_GAMMA = 1.5
DICE_WEIGHT = 0.5
CE_WEIGHT = 0.3
USE_LOVASZ_LOSS = True
LOVASZ_WEIGHT = 0.2
USE_BOUNDARY_LOSS = True
BOUNDARY_WEIGHT = 0.1
BOUNDARY_SIGMA = 2

# ====================== 模型配置 ======================
USE_SE_ATTENTION = True
SE_REDUCTION = 16

# ====================== 数据增强配置 ======================
USE_ELASTIC_TRANSFORM = True
ELASTIC_ALPHA = 800
ELASTIC_SIGMA = 15
USE_CUTOUT = True
CUTOUT_PROB = 0.15
USE_RANDOM_ROTATION = True
ROTATION_RANGE = 8
USE_RANDOM_BRIGHTNESS_CONTRAST = True
BRIGHTNESS_RANGE = 0.1
CONTRAST_RANGE = 0.1

# ====================== 自适应阈值 ======================
USE_ADAPTIVE_THRESHOLD = True
THRESHOLD_SEARCH_STEP = 0.05
THRESHOLD_SEARCH_RANGE = (0.3, 0.8)
USE_OHEM = False
OHEM_RATIO = 0.2
USE_WARM_RESTARTS = True
RESTART_T_0 = 15
RESTART_T_MULT = 2

# ====================== 病灶分类任务配置 ======================
LESION_LABELS = [
    'ICH', 'IPH', 'IVH', 'SDH', 'EDH', 'SAH',
    'Fracture', 'MassEffect', 'MidlineShift'
]
LABEL_COLUMNS = LESION_LABELS
READER_PREFIXES = ['R1:', 'R2:', 'R3:']

# ---------- 2D模式核心配置（MIL + 多窗） ----------
USE_3D_VOLUME = False
USE_THREE_VIEWS = False
INPUT_CHANNELS = 3
NUM_CLASSES = 9
POSITIVE_THRESHOLD = 0.5

# ---------- 目标尺寸 ----------
TARGET_SIZE_2D = (512, 512)
TARGET_SIZE_3D = TARGET_SHAPE_3D

# ---------- 采样配置 ----------
NUM_SLICES_PER_VOLUME = 16
# ========== 候选切片筛选（默认关闭，因实验效果不佳） ==========
CANDIDATE_SELECTION = False          # 关闭，使用均匀采样
CANDIDATE_TOP_K = 16
CANDIDATE_METRICS = ['std', 'edge', 'high_hu_area']

# ---------- 2.5D配置（关闭） ----------
USE_2D5 = False
CONTEXT_SLICES = 1
INPUT_CHANNELS_2D5 = 3 * (1 + 2 * CONTEXT_SLICES)

# ---------- 多窗CT配置 ----------
WINDOW_SETTINGS = {
    'brain': {'wl': 40, 'ww': 80},
    'subdural': {'wl': 50, 'ww': 130},
    'bone': {'wl': 600, 'ww': 2800}
}
WINDOW_NAMES = ['brain', 'subdural', 'bone']

# ---------- MIL配置 ----------
MIL_POOLING = 'topk'
MIL_TOPK = 5
MIL_ATTENTION_DIM = 512

# ---------- 损失配置 ----------
USE_ASL_LOSS = False
ASL_GAMMA_NEG = 4.0
ASL_GAMMA_POS = 1.0
ASL_CLIP = 0.05

# ---------- 阈值优化配置 ----------
USE_PER_CLASS_THRESHOLD = True
THRESHOLD_SEARCH_STEPS = 50
THRESHOLD_MIN = 0.05
THRESHOLD_MAX = 0.95

# ---------- Class-Balanced Loss 配置 ----------
USE_CLASS_BALANCED_LOSS = False
CB_LOSS_BETA = 0.99

# ---------- 训练配置 ----------
BATCH_SIZE_TRAIN = 4
BATCH_SIZE_VAL = 4
NUM_WORKERS = 6
PREFETCH_FACTOR = 2
PIN_MEMORY = True
PERSISTENT_WORKERS = True
USE_AMP = True
GRAD_CLIP_3D = 2.0                      # 提高梯度裁剪
EARLY_STOP_PATIENCE_3D = 25

# ---------- 优化器配置 ----------
OPTIMIZER_3D = "AdamW"
LR_3D = 1e-4                            # 降低学习率（从1.5e-4降至1e-4）
WEIGHT_DECAY_3D = 5e-4
USE_WARMUP = True
WARMUP_EPOCHS = 5

# ====================== 3D训练配置（保留兼容） ======================
MODEL_CHANNELS = [16, 32, 64, 128, 256, 512]
MOMENTUM_3D = 0.9
COSINE_T_MAX = 120
USE_MIXUP = True
MIXUP_ALPHA = 0.2
USE_CUTMIX = True
CUTMIX_ALPHA = 0.2
MIXUP_PROB = 0.3
CUTMIX_PROB = 0.2
GRAD_ACCUM_STEPS = 1
EMA_DECAY = 0.999
USE_EMA = True
USE_TORCH_COMPILE = False
DROPOUT_RATE_3D = 0.4

# ====================== 模型改进开关（实验性，默认关闭） ======================
USE_INDEPENDENT_HEADS = False           # 关闭独立头，使用共享头
PRETRAIN_WEIGHT_PATH = ""               # 留空使用ImageNet

# ====================== 3D训练配置（保留兼容） ======================
MULTI_TASK = False
CLASS_WEIGHTS_3D = [1.0, 1.2, 3.0, 2.0, 10.0, 2.0, 4.0, 1.0, 1.2]
CLASS_WEIGHTS = [1.0, 1.2, 3.0, 2.0, 10.0, 2.0, 4.0, 1.0, 1.2]

# ====================== 训练日志目录 ======================
TRAINING_LOG_DIR = os.path.join(BASE_DIR, "文档", "训练记录")
os.makedirs(TRAINING_LOG_DIR, exist_ok=True)

# ====================== 推理 API 配置 ======================
API_LOG_DIR = os.path.join(BASE_DIR, "logs", "inference")
os.makedirs(API_LOG_DIR, exist_ok=True)

MODEL_VERSIONS = {
    "v1": os.path.join(WEIGHT_SAVE_DIR, "best_lesion.pth"),
    "v2": os.path.join(WEIGHT_SAVE_DIR, "final_best_lesion.pth"),
}
DEFAULT_MODEL_VERSION = "v2"

# LLM 配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-gkhdynknixxmgjuxbjywzdmpacinebnushnlyjvjdstviend")
DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
LLM_CACHE_SIZE = 100
LLM_TIMEOUT = 30
LLM_MAX_RETRIES = 3

# ====================== 环境变量 ======================
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

# ====================== 模型配置（新增） ======================
MODEL_SIZE = "medium"
AUGMENTATION_STRENGTH = "high"

# ====================== 3D训练配置（保留兼容） ======================
USE_OVERSAMPLING = False