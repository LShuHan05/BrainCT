"""
BrainCT推理模块
提供生产级CT伪影分割推理功能
"""

# Inference/__init__.py
from .LesionInferenceAPI import app
from .CTArtifactInfer import CTArtifactInfer