import sys
import os
import SimpleITK as sitk
import numpy as np
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *

import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

# ===================== 这里导入你的模型 =====================
import torch
from ..Model.UNet2D import UNet2D
from .CTArtifactInfer import CTArtifactInfer
# ===========================================================

# ===================== 异步进度条弹窗 =====================
class ProgressDialog(QDialog):
    def __init__(self, title="处理中...", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setFixedSize(450, 120)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 20, 30, 20)
        layout.setSpacing(12)

        self.label = QLabel("初始化处理...")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.progress)

    def set_progress(self, value, text=""):
        self.progress.setValue(value)
        if text:
            self.label.setText(text)

# ===================== 工作线程（异步执行） =====================
class Worker(QThread):
    progress = Signal(int, str)
    finished = Signal()
    error = Signal(str)
    success_msg = Signal(str)

    def __init__(self, func, *args, parent=None):
        super().__init__(parent)
        self.func = func
        self.args = args

    def run(self):
        try:
            self.func(self.report_progress, *self.args)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

    def report_progress(self, value, text=""):
        self.progress.emit(value, text)

# ===================== 主界面 =====================
class ArtifactAnnotationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CT金属伪影掩码标注系统 - 支持DICOM/MHD/NIfTI + VTK 3D视图 + AI自动分割")
        self.resize(1600, 900)

        self.ct_image = None
        self.artifact_mask = None
        self.volume_np = None
        self.mask_np = None

        self.slice_axial = 0
        self.slice_coronal = 0
        self.slice_sagittal = 0

        # ===================== 加载你的 AI 模型 =====================
        self.infer = CTArtifactInfer(model_weight_path="best.pth")
        # ===========================================================

        self.init_ui()
        self.init_vtk_3d_view()

    # -------------------------------------------------------------------------
    # 初始化界面
    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # ========== 左侧控制面板 ==========
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        control_widget.setFixedWidth(340)

        # 加载按钮组
        self.btn_load = QPushButton("加载 DICOM 序列文件夹")
        self.btn_load.clicked.connect(self.load_dicom_async)
        control_layout.addWidget(self.btn_load)

        self.btn_load_mhd = QPushButton("加载 MHD/RAW 三维影像")
        self.btn_load_mhd.clicked.connect(self.load_mhd_async)
        control_layout.addWidget(self.btn_load_mhd)

        self.btn_load_nifti = QPushButton("加载 NIfTI 影像(.nii/.nii.gz)")
        self.btn_load_nifti.clicked.connect(self.load_nifti_async)
        control_layout.addWidget(self.btn_load_nifti)

        self.btn_load_mask = QPushButton("加载掩码文件(nii/nii.gz/mhd)")
        self.btn_load_mask.clicked.connect(self.load_mask_async)
        control_layout.addWidget(self.btn_load_mask)

        self.btn_load_single = QPushButton("加载 单张图像(DCM/PNG)")
        self.btn_load_single.clicked.connect(self.load_single_async)
        control_layout.addWidget(self.btn_load_single)

        # ===================== 【新增】AI 自动分割按钮 =====================
        control_layout.addSpacing(10)
        self.btn_ai_segment = QPushButton("🤖 AI 自动分割 (UNet)")
        self.btn_ai_segment.setStyleSheet("font-weight:bold; color:white; background-color:#2E86AB;")
        self.btn_ai_segment.clicked.connect(self.ai_segment_async)
        self.btn_ai_segment.setEnabled(False)  # 初始禁用
        control_layout.addWidget(self.btn_ai_segment)
        # ==================================================================

        # 阈值下限滑动条
        control_layout.addWidget(QLabel("阈值下限｜二值化阈值滤波"))
        self.th_low = QSlider(Qt.Horizontal)
        self.th_low.setRange(300, 1500)
        self.th_low.setValue(600)
        self.th_low_label = QLabel("600")
        control_layout.addWidget(self.th_low)
        control_layout.addWidget(self.th_low_label)

        # 阈值上限滑动条
        control_layout.addWidget(QLabel("阈值上限｜二值化阈值滤波"))
        self.th_high = QSlider(Qt.Horizontal)
        self.th_high.setRange(2000, 4000)
        self.th_high.setValue(2800)
        self.th_high_label = QLabel("2800")
        control_layout.addWidget(self.th_high)
        control_layout.addWidget(self.th_high_label)

        # 梯度阈值
        control_layout.addWidget(QLabel("梯度阈值｜梯度幅值滤波"))
        self.grad_th = QSlider(Qt.Horizontal)
        self.grad_th.setRange(50, 500)
        self.grad_th.setValue(100)
        self.grad_th_label = QLabel("100")
        control_layout.addWidget(self.grad_th)
        control_layout.addWidget(self.grad_th_label)

        # 形态学开运算半径
        control_layout.addWidget(QLabel("开运算半径｜形态学开运算"))
        self.open_r = QSlider(Qt.Horizontal)
        self.open_r.setRange(0, 5)
        self.open_r.setValue(1)
        self.open_r_label = QLabel("1")
        control_layout.addWidget(self.open_r)
        control_layout.addWidget(self.open_r_label)

        # 形态学闭运算半径
        control_layout.addWidget(QLabel("闭运算半径｜形态学闭运算"))
        self.close_r = QSlider(Qt.Horizontal)
        self.close_r.setRange(0, 10)
        self.close_r.setValue(2)
        self.close_r_label = QLabel("2")
        control_layout.addWidget(self.close_r)
        control_layout.addWidget(self.close_r_label)

        # 最小连通域面积过滤
        control_layout.addWidget(QLabel("最小面积｜连通域过滤"))
        self.min_area = QSlider(Qt.Horizontal)
        self.min_area.setRange(10, 500)
        self.min_area.setValue(40)
        self.min_area_label = QLabel("40")
        control_layout.addWidget(self.min_area)
        control_layout.addWidget(self.min_area_label)

        # 执行与保存按钮
        self.btn_run = QPushButton("生成伪影掩码")
        self.btn_run.clicked.connect(self.generate_mask_async)
        control_layout.addWidget(self.btn_run)

        self.btn_save = QPushButton("保存掩码(NIfTI)")
        self.btn_save.clicked.connect(self.save_mask_async)
        control_layout.addWidget(self.btn_save)

        control_layout.addStretch()
        main_layout.addWidget(control_widget)

        # ========== 右侧整体布局 ==========
        right_layout = QVBoxLayout()

        axial_vtk_layout = QHBoxLayout()

        # 轴位视图
        axial_layout = QVBoxLayout()
        axial_title_layout = QHBoxLayout()
        axial_title_layout.addWidget(QLabel("【轴位 Axial】"))
        axial_title_layout.addStretch()
        self.axial_slice_label = QLabel("切片：0 / 0")
        self.axial_slice_label.setStyleSheet("""
            QLabel { color: #0066CC; font-weight: bold; font-size: 12pt; background-color: #F0F8FF; padding: 4px 8px; border-radius: 4px; }
        """)
        axial_title_layout.addWidget(self.axial_slice_label)
        axial_layout.addLayout(axial_title_layout)

        self.axial_label = QLabel()
        self.axial_label.setStyleSheet("background-color:#111;min-height:280px")
        self.axial_label.setAlignment(Qt.AlignCenter)
        self.slider_axial = QSlider(Qt.Horizontal)
        self.slider_axial.valueChanged.connect(lambda v: self.on_slice_change("axial", v))
        axial_layout.addWidget(self.axial_label)
        axial_layout.addWidget(self.slider_axial)
        axial_vtk_layout.addLayout(axial_layout)

        # VTK 3D 视图
        vtk_box = QVBoxLayout()
        vtk_box.addWidget(QLabel("【VTK 3D CT 重建】"))
        self.vtk_widget = QWidget()
        self.vtk_widget.setStyleSheet("background-color:#000;")
        self.vtk_widget.setMinimumWidth(350)
        vtk_box.addWidget(self.vtk_widget)
        axial_vtk_layout.addLayout(vtk_box)

        right_layout.addLayout(axial_vtk_layout)

        # 冠状 + 矢状 视图并排
        cor_sag_layout = QHBoxLayout()

        # 冠状视图
        coronal_layout = QVBoxLayout()
        coronal_title_layout = QHBoxLayout()
        coronal_title_layout.addWidget(QLabel("【冠状 Coronal】"))
        coronal_title_layout.addStretch()
        self.coronal_slice_label = QLabel("切片：0 / 0")
        self.coronal_slice_label.setStyleSheet("""
            QLabel { color: #0066CC; font-weight: bold; font-size: 12pt; background-color: #F0F8FF; padding: 4px 8px; border-radius: 4px; }
        """)
        coronal_title_layout.addWidget(self.coronal_slice_label)
        coronal_layout.addLayout(coronal_title_layout)

        self.coronal_label = QLabel()
        self.coronal_label.setStyleSheet("background-color:#111;min-height:280px")
        self.coronal_label.setAlignment(Qt.AlignCenter)
        self.slider_coronal = QSlider(Qt.Horizontal)
        self.slider_coronal.valueChanged.connect(lambda v: self.on_slice_change("coronal", v))
        coronal_layout.addWidget(self.coronal_label)
        coronal_layout.addWidget(self.slider_coronal)
        cor_sag_layout.addLayout(coronal_layout)

        # 矢状视图
        sagittal_layout = QVBoxLayout()
        sagittal_title_layout = QHBoxLayout()
        sagittal_title_layout.addWidget(QLabel("【矢状 Sagittal】"))
        sagittal_title_layout.addStretch()
        self.sagittal_slice_label = QLabel("切片：0 / 0")
        self.sagittal_slice_label.setStyleSheet("""
            QLabel { color: #0066CC; font-weight: bold; font-size: 12pt; background-color: #F0F8FF; padding: 4px 8px; border-radius: 4px; }
        """)
        sagittal_title_layout.addWidget(self.sagittal_slice_label)
        sagittal_layout.addLayout(sagittal_title_layout)

        self.sagittal_label = QLabel()
        self.sagittal_label.setStyleSheet("background-color:#111;min-height:280px")
        self.sagittal_label.setAlignment(Qt.AlignCenter)
        self.slider_sagittal = QSlider(Qt.Horizontal)
        self.slider_sagittal.valueChanged.connect(lambda v: self.on_slice_change("sagittal", v))
        sagittal_layout.addWidget(self.sagittal_label)
        sagittal_layout.addWidget(self.slider_sagittal)
        cor_sag_layout.addLayout(sagittal_layout)

        right_layout.addLayout(cor_sag_layout)
        main_layout.addLayout(right_layout)

        self.connect_all_sliders()

    # -------------------------------------------------------------------------
    # VTK 3D 视图初始化
    def init_vtk_3d_view(self):
        self.vtk_view = QVTKRenderWindowInteractor(self.vtk_widget)
        self.vtk_renderer = vtk.vtkRenderer()
        self.vtk_renderer.SetBackground(0.1, 0.1, 0.1)
        self.vtk_view.GetRenderWindow().AddRenderer(self.vtk_renderer)
        self.iren = self.vtk_view.GetRenderWindow().GetInteractor()
        self.iren.Initialize()

        layout = QVBoxLayout(self.vtk_widget)
        layout.addWidget(self.vtk_view)
        layout.setContentsMargins(0,0,0,0)

    # -------------------------------------------------------------------------
    # 3D CT 表面重建（已修复：等值面提取 + 掩码红色3D）
    def render_3d_ct(self):
        if self.ct_image is None:
            return

        from vtkmodules.util import numpy_support
        self.vtk_renderer.RemoveAllViewProps()

        vol = self.volume_np.astype(np.short)
        depth, height, width = vol.shape

        vtk_data = numpy_support.numpy_to_vtk(
            vol.ravel(), deep=True, array_type=vtk.VTK_SHORT
        )

        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(width, height, depth)
        vtk_image.GetPointData().SetScalars(vtk_data)

        # CT 骨骼/高密度 表面重建
        contour = vtk.vtkMarchingCubes()
        contour.SetInputData(vtk_image)
        contour.SetValue(0, 200)
        contour.ComputeNormalsOn()
        contour.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(contour.GetOutputPort())
        mapper.ScalarVisibilityOff()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.9, 0.9, 0.95)
        actor.GetProperty().SetAmbient(0.3)
        actor.GetProperty().SetDiffuse(0.8)
        actor.GetProperty().SetSpecular(0.1)

        self.vtk_renderer.AddActor(actor)

        # 掩码 3D 红色显示
        if self.mask_np is not None:
            mask_vol = self.mask_np.astype(np.uint8)
            mask_vtk = numpy_support.numpy_to_vtk(
                mask_vol.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR
            )

            mask_img = vtk.vtkImageData()
            mask_img.SetDimensions(width, height, depth)
            mask_img.GetPointData().SetScalars(mask_vtk)

            mask_contour = vtk.vtkMarchingCubes()
            mask_contour.SetInputData(mask_img)
            mask_contour.SetValue(0, 0.5)
            mask_contour.Update()

            mask_mapper = vtk.vtkPolyDataMapper()
            mask_mapper.SetInputConnection(mask_contour.GetOutputPort())
            mask_mapper.ScalarVisibilityOff()

            mask_actor = vtk.vtkActor()
            mask_actor.SetMapper(mask_mapper)
            mask_actor.GetProperty().SetColor(1.0, 0.2, 0.2)
            mask_actor.GetProperty().SetOpacity(0.6)

            self.vtk_renderer.AddActor(mask_actor)

        self.vtk_renderer.ResetCamera()
        self.vtk_view.GetRenderWindow().Render()

    # -------------------------------------------------------------------------
    # 绑定滑动条事件
    def connect_all_sliders(self):
        self.th_low.valueChanged.connect(lambda v: self.th_low_label.setText(str(v)))
        self.th_high.valueChanged.connect(lambda v: self.th_high_label.setText(str(v)))
        self.grad_th.valueChanged.connect(lambda v: self.grad_th_label.setText(str(v)))
        self.open_r.valueChanged.connect(lambda v: self.open_r_label.setText(str(v)))
        self.close_r.valueChanged.connect(lambda v: self.close_r_label.setText(str(v)))
        self.min_area.valueChanged.connect(lambda v: self.min_area_label.setText(str(v)))

        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.generate_mask_async)

        self.th_low.valueChanged.connect(lambda: self.update_timer.start(300))
        self.th_high.valueChanged.connect(lambda: self.update_timer.start(300))
        self.grad_th.valueChanged.connect(lambda: self.update_timer.start(300))
        self.open_r.valueChanged.connect(lambda: self.update_timer.start(300))
        self.close_r.valueChanged.connect(lambda: self.update_timer.start(300))
        self.min_area.valueChanged.connect(lambda: self.update_timer.start(300))

    # -------------------------------------------------------------------------
    # 异步任务封装
    def run_async(self, title, func, *args):
        self.dialog = ProgressDialog(title, self)
        self.worker = Worker(func, *args)
        self.worker.progress.connect(self.dialog.set_progress)
        self.worker.finished.connect(self.dialog.close)
        self.worker.finished.connect(self.on_task_done)
        self.worker.error.connect(lambda e: (self.dialog.close(), QMessageBox.critical(self, "错误", e)))
        self.worker.success_msg.connect(self.on_success_msg)
        self.dialog.show()
        self.worker.start()

    def on_task_done(self):
        self.refresh_all_views()
        self.update_all_slice_labels()
        self.render_3d_ct()

    def on_success_msg(self, msg):
        QMessageBox.information(self, "成功", msg)

    # -------------------------------------------------------------------------
    # 【新增】AI 分割（调用你的 UNet）
    def ai_segment_async(self):
        if self.ct_image is None:
            QMessageBox.warning(self, "提示", "请先加载CT图像！")
            return
        self.run_async("AI 模型分割中...", self._ai_segment_task)

    def _ai_segment_task(self, report):
        report(30, "模型推理中...")
        # ===================== 调用你的模型 =====================
        sitk_mask = self.infer.predict_from_sitk(self.ct_image)
        # =======================================================
        report(80, "生成掩码...")
        self.artifact_mask = sitk_mask
        self.mask_np = sitk.GetArrayFromImage(sitk_mask)
        report(100, "AI 分割完成 ✅")

    # -------------------------------------------------------------------------
    # 加载 MHD
    def load_mhd_async(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 MHD 文件", "", "MHD Image (*.mhd *.MHD)")
        if not path: return
        self.run_async("加载 MHD 中...", self._load_mhd_task, path)

    def _load_mhd_task(self, report, path):
        report(20, "读取MHD文件...")
        img = sitk.ReadImage(path)
        report(60, "解析三维数据...")
        vol = sitk.GetArrayFromImage(img)
        self.ct_image = img
        self.volume_np = vol
        self.mask_np = None
        self.artifact_mask = None
        self.init_slice_range()
        report(100, "完成")
        self.btn_ai_segment.setEnabled(True)  # 启用 AI 按钮

    # -------------------------------------------------------------------------
    # 加载 DICOM
    def load_dicom_async(self):
        folder = QFileDialog.getExistingDirectory(self, "选择 DICOM 序列文件夹")
        if not folder: return
        self.run_async("加载 DICOM 中...", self._load_dicom_task, folder)

    def _load_dicom_task(self, report, folder):
        report(10, "扫描序列文件...")
        reader = sitk.ImageSeriesReader()
        fns = reader.GetGDCMSeriesFileNames(folder)
        reader.SetFileNames(fns)
        report(40, "读取图像...")
        img = reader.Execute()
        report(70, "构建三维体...")
        vol = sitk.GetArrayFromImage(img)
        self.ct_image = img
        self.volume_np = vol
        self.mask_np = None
        self.artifact_mask = None
        self.init_slice_range()
        report(100, "完成")
        self.btn_ai_segment.setEnabled(True)

    # -------------------------------------------------------------------------
    # 加载单张
    def load_single_async(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图像", "", "所有图像 (*.dcm *.png *.jpg *.nii *.mhd)")
        if not path: return
        self.run_async("加载单张图像...", self._load_single_task, path)

    def _load_single_task(self, report, path):
        img = sitk.ReadImage(path)
        vol = sitk.GetArrayFromImage(img)
        if vol.ndim == 2:
            vol = vol[None]
            img = sitk.GetImageFromArray(vol)
        self.ct_image = img
        self.volume_np = vol
        self.mask_np = None
        self.artifact_mask = None
        self.init_slice_range()
        report(100)
        self.btn_ai_segment.setEnabled(True)

    # -------------------------------------------------------------------------
    # 加载 NIfTI
    def load_nifti_async(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 NIfTI 影像文件", "",
            "NIfTI Image (*.nii *.nii.gz *.NII *.NII.GZ)"
        )
        if not path: return
        self.run_async("加载 NIfTI 中...", self._load_nifti_task, path)

    def _load_nifti_task(self, report, path):
        report(20, "读取 NIfTI 文件...")
        img = sitk.ReadImage(path)
        report(60, "解析体数据...")
        vol = sitk.GetArrayFromImage(img)
        self.ct_image = img
        self.volume_np = vol
        self.mask_np = None
        self.artifact_mask = None
        self.init_slice_range()
        report(100, "加载完成")
        self.btn_ai_segment.setEnabled(True)

    # -------------------------------------------------------------------------
    # 加载掩码文件
    def load_mask_async(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择掩码文件", "",
            "Mask Files (*.nii *.nii.gz *.mhd *.MHD)"
        )
        if not path: return
        self.run_async("加载掩码中...", self._load_mask_task, path)

    def _load_mask_task(self, report, path):
        if self.ct_image is None:
            raise Exception("请先加载CT图像！")

        report(20, "读取掩码文件...")
        mask_img = sitk.ReadImage(path)

        if mask_img.GetSize() != self.ct_image.GetSize():
            raise Exception("掩码尺寸与CT不匹配！")

        report(60, "转换掩码数据...")
        mask_np = sitk.GetArrayFromImage(mask_img)
        self.artifact_mask = mask_img
        self.mask_np = (mask_np > 0).astype(np.uint8)

        report(100, "掩码加载完成 ✅")

    # -------------------------------------------------------------------------
    # 生成伪影掩码
    def generate_mask_async(self):
        if self.ct_image is None: return
        self.run_async("生成伪影掩码...", self._generate_mask_task)

    def _generate_mask_task(self, report):
        img = self.ct_image
        tl = self.th_low.value()
        th = self.th_high.value()
        gg = self.grad_th.value()
        ro = self.open_r.value()
        rc = self.close_r.value()
        ma = self.min_area.value()

        report(10, "二值化阈值处理...")
        mask = sitk.BinaryThreshold(img, tl, th, 1, 0)

        report(25, "计算梯度幅值...")
        g = sitk.GradientMagnitude(sitk.Cast(img, sitk.sitkFloat32))
        gmask = sitk.BinaryThreshold(g, gg, 99999, 1, 0)

        comb = sitk.And(mask, gmask)

        report(40, "形态学开运算...")
        comb = sitk.BinaryMorphologicalOpening(comb, [ro, ro, ro])

        report(55, "形态学闭运算...")
        comb = sitk.BinaryMorphologicalClosing(comb, [rc, rc, rc])

        report(70, "连通域分析...")
        cc = sitk.ConnectedComponent(comb)
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(cc)

        report(85, "生成最终掩码...")
        final = sitk.Image(img.GetSize(), sitk.sitkUInt8)
        final.CopyInformation(img)
        for lbl in stats.GetLabels():
            if stats.GetNumberOfPixels(lbl) >= ma:
                final += sitk.BinaryThreshold(cc, lbl, lbl, 1, 0)

        self.artifact_mask = final
        self.mask_np = sitk.GetArrayFromImage(final)
        report(100, "处理完成")

    # -------------------------------------------------------------------------
    # 保存掩码
    def save_mask_async(self):
        if self.artifact_mask is None:
            QMessageBox.warning(self, "提示", "先生成掩码！")
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存掩码", "artifact_mask.nii.gz", "NIfTI 掩码 (*.nii.gz *.nii)")
        if not path: return
        if os.path.exists(path):
            yes = QMessageBox.question(self, "文件已存在", f"{path}\n已存在，是否覆盖？")
            if yes != QMessageBox.Yes:
                return
        self.run_async("保存掩码中...", self._save_mask_task, path)

    def _save_mask_task(self, report, path):
        sitk.WriteImage(self.artifact_mask, path)
        report(100, "保存完成")
        self.worker.success_msg.emit(f"掩码已保存到：\n{path}")

    # -------------------------------------------------------------------------
    # 切片显示控制
    def update_all_slice_labels(self):
        if self.volume_np is None: return
        d, h, w = self.volume_np.shape
        self.axial_slice_label.setText(f"切片：{self.slice_axial} / {d-1}")
        self.coronal_slice_label.setText(f"切片：{self.slice_coronal} / {h-1}")
        self.sagittal_slice_label.setText(f"切片：{self.slice_sagittal} / {w-1}")

    def init_slice_range(self):
        d, h, w = self.volume_np.shape
        self.slider_axial.setRange(0, d-1)
        self.slider_coronal.setRange(0, h-1)
        self.slider_sagittal.setRange(0, w-1)
        sa, sc, ss = d//2, h//2, w//2
        self.slice_axial, self.slice_coronal, self.slice_sagittal = sa, sc, ss
        self.slider_axial.setValue(sa)
        self.slider_coronal.setValue(sc)
        self.slider_sagittal.setValue(ss)
        self.update_all_slice_labels()

    # -------------------------------------------------------------------------
    # 三平面同步滑动
    def on_slice_change(self, vt, v):
        if self.volume_np is None:
            return

        d, h, w = self.volume_np.shape

        if vt == "axial":
            self.slice_axial = v
            self.slice_coronal = int((v / d) * h)
            self.slice_sagittal = int((v / d) * w)

        elif vt == "coronal":
            self.slice_coronal = v
            self.slice_axial = int((v / h) * d)
            self.slice_sagittal = int((v / h) * w)

        elif vt == "sagittal":
            self.slice_sagittal = v
            self.slice_axial = int((v / w) * d)
            self.slice_coronal = int((v / w) * h)

        self.slice_axial = np.clip(self.slice_axial, 0, d-1)
        self.slice_coronal = np.clip(self.slice_coronal, 0, h-1)
        self.slice_sagittal = np.clip(self.slice_sagittal, 0, w-1)

        self.slider_axial.blockSignals(True)
        self.slider_axial.setValue(self.slice_axial)
        self.slider_axial.blockSignals(False)

        self.slider_coronal.blockSignals(True)
        self.slider_coronal.setValue(self.slice_coronal)
        self.slider_coronal.blockSignals(False)

        self.slider_sagittal.blockSignals(True)
        self.slider_sagittal.setValue(self.slice_sagittal)
        self.slider_sagittal.blockSignals(False)

        self.refresh_all_views()
        self.update_all_slice_labels()

    # -------------------------------------------------------------------------
    # 切片绘制
    def draw_slice(self, im, mk):
        im = np.clip(im, -500, 1500)
        im = ((im - im.min()) / (im.max() - im.min() + 1e-8) * 255).astype(np.uint8)
        c = np.stack([im, im, im], axis=-1)
        if mk is not None and mk.shape == im.shape:
            c[mk > 0] = [255, 80, 80]
        return c

    def to_qpix(self, arr):
        h, w, ch = arr.shape
        qim = QImage(arr.data, w, h, 3*w, QImage.Format_RGB888)
        return QPixmap.fromImage(qim).scaled(500, 500, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    # -------------------------------------------------------------------------
    # 刷新视图
    def update_view_axial(self):
        if self.volume_np is None: return
        im = self.volume_np[self.slice_axial]
        mk = self.mask_np[self.slice_axial] if self.mask_np is not None else None
        self.axial_label.setPixmap(self.to_qpix(self.draw_slice(im, mk)))

    def update_view_coronal(self):
        if self.volume_np is None: return
        im = self.volume_np[:, self.slice_coronal, :]
        mk = self.mask_np[:, self.slice_coronal, :] if self.mask_np is not None else None
        self.coronal_label.setPixmap(self.to_qpix(self.draw_slice(im, mk)))

    def update_view_sagittal(self):
        if self.volume_np is None: return
        im = self.volume_np[:, :, self.slice_sagittal]
        mk = self.mask_np[:, :, self.slice_sagittal] if self.mask_np is not None else None
        self.sagittal_label.setPixmap(self.to_qpix(self.draw_slice(im, mk)))

    def refresh_all_views(self):
        self.update_view_axial()
        self.update_view_coronal()
        self.update_view_sagittal()

# -------------------------------------------------------------------------
# 主程序入口
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ArtifactAnnotationWindow()
    win.show()
    sys.exit(app.exec())