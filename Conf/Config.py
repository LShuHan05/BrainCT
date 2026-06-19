import os
import torch

# ====================== 路径配置 ======================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

SAVE_DIR = os.path.join(BASE_DIR, "run")
WEIGHT_SAVE_DIR = os.path.join(SAVE_DIR, "weights")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(WEIGHT_SAVE_DIR, exist_ok=True)
WEIGHT_FILE = "best.pth"

# 原始数据路径（保留用于伪影分割）
CT_PATH = os.path.join(BASE_DIR, "datasets", "CT")
MASK_PATH = os.path.join(BASE_DIR, "datasets", "MASK")

# CQ500 病灶检测数据路径
CQ500_DATA_ROOT = os.path.join(BASE_DIR, "datasets_filtered")
CQ500_ANNOTATION_FILE = os.path.join(BASE_DIR, "md", "reads.csv")

# ====================== 任务模式选择 ======================
TASK_MODE = "lesion_classification"  # "artifact_segmentation" 或 "lesion_classification"

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
USE_AMP = True
GRAD_CLIP = 1.0
EARLY_STOP_PATIENCE = 25

# ====================== 【新】损失函数配置 ======================
FOCAL_ALPHA = 0.75
FOCAL_GAMMA = 1.5

DICE_WEIGHT = 0.5
CE_WEIGHT = 0.3

USE_LOVASZ_LOSS = True
LOVASZ_WEIGHT = 0.2

USE_BOUNDARY_LOSS = True
BOUNDARY_WEIGHT = 0.1
BOUNDARY_SIGMA = 2

# ====================== 【新】模型配置 ======================
USE_SE_ATTENTION = True
SE_REDUCTION = 16

# ====================== 【新】数据增强配置 ======================
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

# ====================== 【新增】病灶分类任务配置 ======================
# 标签定义
LESION_LABELS = [
    'ICH', 'IPH', 'IVH', 'SDH', 'EDH', 'SAH',
    'Fracture', 'MassEffect', 'MidlineShift'
]
NUM_CLASSES = len(LESION_LABELS)

# 模型输入模式
USE_3D_VOLUME = False  # True=3D体积, False=2D切片
USE_THREE_VIEWS = True  # 是否使用三视图（轴位+冠状+矢状）
INPUT_CHANNELS = 3 if USE_THREE_VIEWS else 1

# 目标尺寸
TARGET_SIZE_2D = (256, 256)  # 2D切片尺寸
TARGET_SIZE_3D = (128, 128, 64)  # 3D体积尺寸 (D, H, W)

# 多标签分类损失权重（可调整类别不平衡）
CLASS_WEIGHTS = [1.0] * NUM_CLASSES  # 可根据数据分布调整

# 评估阈值
POSITIVE_THRESHOLD = 0.5

# 抽样配置
SAMPLE_RATIO_POSITIVE = 0.6  # 阳性样本比例
TOTAL_SAMPLES = 100  # 总抽样数量

# ====================== 【新增】训练日志目录 ======================
TRAINING_LOG_DIR = os.path.join(BASE_DIR, "文档", "训练记录")
os.makedirs(TRAINING_LOG_DIR, exist_ok=True)

# ====================== 推理 API 配置 ======================
API_LOG_DIR = os.path.join(BASE_DIR, "logs", "inference")
os.makedirs(API_LOG_DIR, exist_ok=True)

# 模型版本（可加载多个）
MODEL_VERSIONS = {
    "v1": os.path.join(WEIGHT_SAVE_DIR, "best_lesion.pth"),
    # 后续可继续添加 v2, v3...
}
DEFAULT_MODEL_VERSION = "v1"

# LLM 配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-gkhdynknixxmgjuxbjywzdmpacinebnushnlyjvjdstviend")
DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
LLM_CACHE_SIZE = 100  # 缓存最近 N 个结果
LLM_TIMEOUT = 30       # 超时时间（秒）
LLM_MAX_RETRIES = 3    # 最大重试次数