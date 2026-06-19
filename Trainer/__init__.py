import sys
import os

# 确保项目根目录在 sys.path 中
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Trainer.UNet2DTrainer import Trainer
from Trainer.LesionClassifierTrainer import LesionClassifierTrainer

__all__ = [
    'Trainer',
    'LesionClassifierTrainer'
]
