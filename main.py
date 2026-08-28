# main.py

import sys
import os

# 确保 code_revise 目录在最前面，优先加载修改版模块
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import json
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGridLayout, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QSlider, QLabel, QComboBox, QListWidget, QListWidgetItem,
    QFrame, QCheckBox, QStackedWidget, QMessageBox, QProgressDialog,
    QDialog, QLineEdit, QFormLayout, QDialogButtonBox,
    QTreeWidget, QTreeWidgetItem, QHeaderView
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from core.engine import NiiEngine
from ui.canvas import MriCanvas, Freeview3DWidget


def get_sample_data_path():
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, "sample_data")
    else:
        return os.path.join(os.path.abspath("."), "sample_data")


def get_mri_data_path():
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, "mri")
    else:
        return os.path.join(os.path.abspath("."), "mri")


class RegistrationWorker(QThread):
    finished = pyqtSignal(bool, str, list)

    def __init__(self, fixed_t1, template_t1, overlays, out_dir, parent=None):
        super().__init__(parent)
        self.fixed_t1 = fixed_t1
        self.template_t1 = template_t1
        self.overlays = overlays
        self.out_dir = out_dir

    def run(self):
        try:
            from pathlib import Path
            import tools.register_template_to_individual as reg

            out_dir = Path(self.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            transform = reg.register_template_to_individual(self.fixed_t1, self.template_t1)
            reg.write_transform(transform, out_dir / "template_to_individual_transform.tfm")

            outputs = []
            for overlay in self.overlays:
                overlay_path = Path(overlay)
                suffix = ".nii.gz" if overlay_path.name.endswith(".nii.gz") else overlay_path.suffix
                stem = overlay_path.name[:-7] if overlay_path.name.endswith(".nii.gz") else overlay_path.stem
                output = out_dir / f"{stem}_in_individual_space{suffix}"
                reg.resample_to_fixed(
                    overlay_path,
                    self.fixed_t1,
                    transform,
                    output,
                    is_label=reg.looks_like_label(overlay_path),
                )
                outputs.append(str(output))
            self.finished.emit(True, f"仿真完成，已生成 {len(outputs)} 个个体空间图层。", outputs)
        except ImportError as exc:
            self.finished.emit(
                False,
                f"缺少仿真依赖: {exc}\n\n请先安装 SimpleITK:\npy -3 -m pip install SimpleITK",
                [],
            )
        except Exception as exc:
            self.finished.emit(False, f"仿真失败:\n{exc}", [])


class ZjuMriViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.engine = NiiEngine()
        self.sliders = {}
        self.current_base_path = None

        self.timer_2d = QTimer(self)
        self.timer_2d.setSingleShot(True)
        self.timer_2d.timeout.connect(self.update_2d_views)

        self.timer_3d = QTimer(self)
        self.timer_3d.setSingleShot(True)
        self.timer_3d.timeout.connect(self.update_3d_view)

        self._drag_active = False
        self._last_browse_dir = None

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("NeuroGuide")
        self.resize(1380, 850)

        self.setStyleSheet("""
            QMainWindow { background-color: #0B0E14; }
            QWidget { color: #CDD6F4; font-family: "Segoe UI", "Microsoft YaHei", sans-serif; }
            QFrame#ctrlPanel { background-color: #111318; border-radius: 12px; border: 1px solid #2A2E38; }
            QPushButton { background-color: #222530; color: #CDD6F4; border-radius: 6px; padding: 6px 12px; font-weight: bold; border: 1px solid #2A2E38; }
            QPushButton:hover { background-color: #2D3140; border-color: #3B82F6; }
            QPushButton#primaryBtn { background-color: #1E66F5; color: white; border: none; }
            QPushButton#primaryBtn:hover { background-color: #3B82F6; }
            QPushButton#stdBrainBtn { background-color: #A580FF; color: white; border: none; }
            QPushButton#stdBrainBtn:hover { background-color: #B899FF; }
            QPushButton#stdBrainBtn:disabled { background-color: #555; color: #999; }
            QPushButton#individualModeBtn { background-color: #1E66F5; color: white; border: none; }
            QPushButton#individualModeBtn:hover { background-color: #3B82F6; }
            QListWidget { background-color: #181B22; border: 1px solid #2A2E38; border-radius: 8px; color: #CDD6F4; outline: none; }
            QListWidget::item { padding: 4px 8px; border-radius: 4px; }
            QListWidget::item:selected { background-color: #1E66F5; }
            QComboBox { background-color: #1E2028; border: 1px solid #2A2E38; border-radius: 6px; padding: 4px 10px; color: #CDD6F4; }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView { background-color: #181B22; border: 1px solid #2A2E38; selection-background-color: #1E66F5; color: #CDD6F4; }
            QSlider::groove:horizontal { background: #2A2E38; height: 4px; border-radius: 2px; }
            QSlider::handle:horizontal { background: #1E66F5; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }
            QCheckBox { spacing: 6px; color: #CDD6F4; }
            QCheckBox::indicator { width: 16px; height: 16px; border-radius: 4px; border: 1px solid #2A2E38; background-color: #0B0E14; }
            QCheckBox::indicator:checked { background-color: #1E66F5; border-color: #1E66F5; }
            QLabel { color: #A0A8C0; font-size: 12px; }
            QFrame#stepFrame {
                background-color: #1A1D26;
                border-radius: 8px;
                border: 1px solid #2A2E38;
                margin-top: 6px;
                margin-bottom: 6px;
            }
            QFrame#stepItem {
                background-color: #222530;
                border-radius: 6px;
                border: 1px solid #2A2E38;
                margin: 2px;
            }
            QFrame#stepItem:hover {
                border-color: #3B82F6;
                background-color: #2D3140;
            }
        """)

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.home_page = self.create_home_page()
        self.viewer_page = self.create_viewer_page()
        self.patient_page = self.create_patient_records_page()

        self.stack.addWidget(self.home_page)
        self.stack.addWidget(self.viewer_page)
        self.stack.addWidget(self.patient_page)
        self.stack.setCurrentIndex(0)

    def show_home(self):
        self.stack.setCurrentIndex(0)

    def create_home_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title = QLabel("NeuroGuide")
        title.setStyleSheet("color: white; font-size: 32px; font-weight: bold;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("经颅电流刺激仿真可视化")
        subtitle.setStyleSheet("color: #A0A8C0; font-size: 14px; margin-bottom: 30px;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

        btn_viewer = QPushButton("进入可视化工作站")
        btn_viewer.setObjectName("primaryBtn")
        btn_viewer.setFixedSize(320, 70)
        btn_viewer.setStyleSheet("font-size: 16px; border-radius: 10px;")
        btn_viewer.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        layout.addWidget(btn_viewer, alignment=Qt.AlignmentFlag.AlignCenter)

        layout.addSpacing(15)

        btn_patients = QPushButton("患者记录")
        btn_patients.setObjectName("primaryBtn")
        btn_patients.setFixedSize(320, 55)
        btn_patients.setStyleSheet("font-size: 14px; border-radius: 10px; background-color: #A580FF; color: white;")
        btn_patients.clicked.connect(lambda: self.stack.setCurrentIndex(2))
        layout.addWidget(btn_patients, alignment=Qt.AlignmentFlag.AlignCenter)
        return page

    def add_ui_divider(self, parent_layout):
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setStyleSheet("background-color: #2A2E38; max-height: 1px;")
        parent_layout.addWidget(line)

    def create_individual_steps_panel(self):
        """创建个体精准模式的五个步骤框（纯UI，无功能）"""
        frame = QFrame()
        frame.setObjectName("stepFrame")
        frame.setVisible(False)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(8)

        header = QHBoxLayout()
        title = QLabel("个体精准模式 · 规划流程")
        title.setStyleSheet("color: #F9F9FA; font-weight: bold; font-size: 14px;")
        note = QLabel("流程界面展示")
        note.setStyleSheet("color: #7A8499; font-size: 11px;")
        header.addWidget(title)
        header.addStretch()
        header.addWidget(note)
        layout.addLayout(header)

        steps = [
            ("1", "AI辅助分割"),
            ("2", "候选电极放置"),
            ("3", "离散网格划分"),
            ("4", "有限元计算"),
            ("5", "智能算法优化")
        ]

        steps_layout = QHBoxLayout()
        steps_layout.setSpacing(8)
        for index, (num, name) in enumerate(steps):
            step_card = QFrame()
            step_card.setObjectName("stepItem")
            step_card.setMinimumHeight(58)
            card_layout = QVBoxLayout(step_card)
            card_layout.setContentsMargins(10, 7, 10, 7)
            card_layout.setSpacing(3)

            idx_label = QLabel(num)
            idx_label.setFixedSize(22, 22)
            idx_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            idx_label.setStyleSheet("""
                background-color: #1E66F5;
                color: white;
                border-radius: 11px;
                font-weight: bold;
                font-size: 11px;
            """)
            step_name = QLabel(name)
            step_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
            step_name.setStyleSheet("font-weight: bold; color: #F9F9FA; font-size: 12px;")
            card_layout.addWidget(idx_label, alignment=Qt.AlignmentFlag.AlignCenter)
            card_layout.addWidget(step_name)
            steps_layout.addWidget(step_card, 1)

            if index < len(steps) - 1:
                arrow = QLabel("->")
                arrow.setAlignment(Qt.AlignmentFlag.AlignCenter)
                arrow.setStyleSheet("color: #3B82F6; font-size: 16px;")
                steps_layout.addWidget(arrow)

        layout.addLayout(steps_layout)

        return frame

    def create_viewer_page(self):
        viewer = QWidget()
        main_layout = QVBoxLayout(viewer)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        self.steps_frame = self.create_individual_steps_panel()
        main_layout.addWidget(self.steps_frame)

        workspace_layout = QHBoxLayout()
        workspace_layout.setSpacing(8)

        view_container = QFrame()
        view_layout = QGridLayout(view_container)
        view_layout.setContentsMargins(0, 0, 0, 0)
        self.canvases = {
            "sagittal": MriCanvas("Sagittal / 矢状面", "sagittal"),
            "coronal": MriCanvas("Coronal / 冠状面", "coronal"),
            "axial": MriCanvas("Axial / 轴状面", "axial")
        }
        self.canvas_3d = Freeview3DWidget()
        self.canvas_3d.slice_changed.connect(self.handle_3d_slider_change)
        for c in self.canvases.values():
            c.clicked.connect(self.handle_canvas_click)
            c.drag_moved.connect(self.handle_canvas_drag)
            c.drag_released.connect(self.handle_canvas_drag_release)

        view_layout.addWidget(self.canvases["sagittal"], 0, 0)
        view_layout.addWidget(self.canvases["coronal"], 0, 1)
        view_layout.addWidget(self.canvases["axial"], 1, 0)
        view_layout.addWidget(self.canvas_3d, 1, 1)

        ctrl_container = QFrame()
        ctrl_container.setObjectName("ctrlPanel")
        ctrl_container.setMinimumWidth(360)
        ctrl_container.setFixedWidth(360)

        ctrl_panel = QVBoxLayout(ctrl_container)
        ctrl_panel.setContentsMargins(12, 12, 12, 12)
        ctrl_panel.setSpacing(6)

        ctrl_panel.addWidget(QLabel("<b style='color:#A580FF; font-size:13px;'>模式选择</b>"))
        mode_layout = QHBoxLayout()
        self.btn_std_mode = QPushButton("标准高效模式")
        self.btn_std_mode.setObjectName("stdBrainBtn")
        self.btn_std_mode.setMinimumHeight(38)
        self.btn_std_mode.setToolTip("加载标准人脑模板，进入标准工作流程")
        self.btn_std_mode.clicked.connect(self.on_standard_brain_clicked)

        self.btn_individual_mode = QPushButton("个体精准模式")
        self.btn_individual_mode.setObjectName("individualModeBtn")
        self.btn_individual_mode.setMinimumHeight(38)
        self.btn_individual_mode.setToolTip("展示个体精准模式五步流程界面")
        self.btn_individual_mode.clicked.connect(self.on_individual_mode_clicked)
        mode_layout.addWidget(self.btn_std_mode, 1)
        mode_layout.addWidget(self.btn_individual_mode, 1)
        ctrl_panel.addLayout(mode_layout)

        self.add_ui_divider(ctrl_panel)

        # 【1】底图加载 — 默认隐藏，点击"个体精准模式"后显示
        self.base_map_section = QFrame()
        self.base_map_section.setVisible(False)
        base_section_layout = QVBoxLayout(self.base_map_section)
        base_section_layout.setContentsMargins(0, 0, 0, 0)
        base_section_layout.setSpacing(6)

        base_section_layout.addWidget(QLabel("<b style='color:#3B82F6; font-size:13px;'>【1】解剖底图加载</b>"))

        mri_base_layout = QHBoxLayout()
        self.mri_combo = QComboBox()
        self.mri_combo.addItem("快速选择本地目录 MRI...")
        self.scan_local_mri_files()
        self.mri_combo.currentIndexChanged.connect(self.on_local_mri_changed)
        mri_base_layout.addWidget(self.mri_combo, 1)

        btn_browse_mri = QPushButton("浏览…")
        btn_browse_mri.setFixedWidth(60)
        btn_browse_mri.setToolTip("从文件管理器选择 NIfTI 文件加载")
        btn_browse_mri.clicked.connect(self.on_browse_mri_clicked)
        mri_base_layout.addWidget(btn_browse_mri, 0)

        base_section_layout.addLayout(mri_base_layout)

        ctrl_panel.addWidget(self.base_map_section)

        self.add_ui_divider(ctrl_panel)

        # 【2】靶点与图谱
        ctrl_panel.addWidget(QLabel("<b style='color:#A580FF; font-size:13px;'>【2】刺激靶点与图谱配置</b>"))

        disease_layout = QHBoxLayout()
        self.disease_combo = QComboBox()
        self.disease_combo.addItem("选择临床疾病...")
        self.disease_combo.currentIndexChanged.connect(self.on_disease_changed)
        self.target_combo = QComboBox()
        self.target_combo.addItem("选择干预靶点...")
        self.target_combo.setEnabled(False)
        self.target_combo.currentIndexChanged.connect(self.on_target_changed)
        disease_layout.addWidget(self.disease_combo, 1)
        disease_layout.addWidget(self.target_combo, 1)
        ctrl_panel.addLayout(disease_layout)
        self.refresh_disease_list()

        overlay_actions_layout = QHBoxLayout()
        self.btn_aal = QPushButton("加载 AAL 脑区图谱")
        self.btn_aal.setObjectName("primaryBtn")
        self.btn_aal.clicked.connect(self.on_load_aal_clicked)
        btn_browse_overlay = QPushButton("手动追加叠加层")
        btn_browse_overlay.setToolTip("手动追加其他彩色衍生层")
        btn_browse_overlay.clicked.connect(self.on_browse_overlay_clicked)
        overlay_actions_layout.addWidget(self.btn_aal, 1)
        overlay_actions_layout.addWidget(btn_browse_overlay, 1)
        ctrl_panel.addLayout(overlay_actions_layout)

        self.btn_register_target = QPushButton("进行个体T1当前靶点仿真")
        self.btn_register_target.setToolTip("仿真得到当前靶点的模板空间灰/白质层并配准到当前加载的个体 T1上")
        self.btn_register_target.clicked.connect(self.on_register_current_target_clicked)
        ctrl_panel.addWidget(self.btn_register_target)

        self.add_ui_divider(ctrl_panel)

        # 【3】系统高性能仿真迭代
        sim_label = QLabel("<b style='color:#A580FF; font-size:13px;'>【3】系统高性能仿真迭代</b>")
        sim_label.setMaximumHeight(24)
        ctrl_panel.addWidget(sim_label)

        sim_btn_layout = QHBoxLayout()
        sim_btn_layout.setSpacing(6)

        btn_report = QPushButton("导出仿真报告")
        btn_report.setObjectName("primaryBtn")
        btn_report.setMinimumHeight(32)
        btn_report.setToolTip("将当前可视化状态导出为 PDF 仿真报告")
        btn_report.clicked.connect(self.on_export_report_clicked)
        sim_btn_layout.addWidget(btn_report, 1)

        btn_electrode = QPushButton("查看 10-10 电极图")
        btn_electrode.setStyleSheet("background-color: #A580FF; color: white;")
        btn_electrode.setMinimumHeight(32)
        btn_electrode.setToolTip("打开 10-10 电极位置图\n电极图片存放位置: assets/ 目录\n请将10-10电极图放入 assets/ 并命名为 electrode_10_10.png")
        btn_electrode.clicked.connect(self.on_electrode_map_clicked)
        sim_btn_layout.addWidget(btn_electrode, 1)

        ctrl_panel.addLayout(sim_btn_layout)

        self.add_ui_divider(ctrl_panel)

        # 公共控制区
        ctrl_panel.addWidget(QLabel("图层管理器"))
        self.layer_list = QListWidget()
        self.layer_list.setMinimumHeight(180)
        self.layer_list.currentRowChanged.connect(self.sync_cmap_selector)
        self.layer_list.itemChanged.connect(self.schedule_update)
        ctrl_panel.addWidget(self.layer_list, 1)

        # ── 一键定位至靶点按钮 ──
        self.btn_goto_target = QPushButton(" 一键定位至靶点")
        self.btn_goto_target.setStyleSheet(
            "background-color: #1E66F5; color: white; font-size: 12px; "
            "padding: 5px 0px; border-radius: 6px; font-weight: bold;")
        self.btn_goto_target.setToolTip("将当前视图中心立即移动到已选靶点的 MNI152 标准坐标")
        self.btn_goto_target.clicked.connect(self.on_goto_target_clicked)
        ctrl_panel.addWidget(self.btn_goto_target, 0)

        self.mask_panel = QFrame()
        self.mask_panel.setVisible(False)
        mask_v = QVBoxLayout(self.mask_panel)
        mask_v.setContentsMargins(0, 0, 0, 0)
        mask_v.setSpacing(4)
        self.chk_apply_mask = QCheckBox("使用此图谱裁剪其他图层")
        self.chk_apply_mask.stateChanged.connect(self.schedule_update)
        mask_v.addWidget(self.chk_apply_mask)
        self.label_list = QListWidget()
        self.label_list.setFixedHeight(70)
        self.label_list.itemChanged.connect(self.on_label_selection_changed)
        mask_v.addWidget(self.label_list)
        ctrl_panel.addWidget(self.mask_panel)

        render_ctrl = QHBoxLayout()
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(["hot", "cool", "jet", "viridis", "gray", "plasma"])
        self.cmap_combo.currentIndexChanged.connect(self.on_cmap_changed)
        self.alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(70)
        self.alpha_slider.valueChanged.connect(self.on_alpha_changed)
        render_ctrl.addWidget(QLabel("色表:"), 0)
        render_ctrl.addWidget(self.cmap_combo, 1)
        render_ctrl.addWidget(QLabel("透明度:"), 0)
        render_ctrl.addWidget(self.alpha_slider, 1)
        ctrl_panel.addLayout(render_ctrl)

        threshold_layout = QHBoxLayout()
        self.min_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_slider.setRange(0, 100)
        self.min_slider.setValue(0)
        self.min_slider.valueChanged.connect(self.schedule_update)
        threshold_layout.addWidget(QLabel("阈值:"), 0)
        threshold_layout.addWidget(self.min_slider, 1)
        ctrl_panel.addLayout(threshold_layout)

        self.info_panel = QFrame()
        self.info_panel.setStyleSheet("background-color: #1A1D26; border: 1px solid #1E66F5; border-radius: 6px;")
        info_vbox = QVBoxLayout(self.info_panel)
        info_vbox.setContentsMargins(8, 4, 8, 4)
        self.coord_label = QLabel("MNI: (0.0, 0.0, 0.0)")
        self.coord_label.setStyleSheet("color: #00FF00; font-family: Consolas; font-size: 11px;")
        info_vbox.addWidget(self.coord_label)
        # ── 电场强度信息：每个叠加层独占一行 ──
        self.val_labels_container = QWidget()
        self.val_labels_layout = QVBoxLayout(self.val_labels_container)
        self.val_labels_layout.setContentsMargins(0, 0, 0, 0)
        self.val_labels_layout.setSpacing(1)
        self.val_labels = []
        info_vbox.addWidget(self.val_labels_container)
        ctrl_panel.addWidget(self.info_panel)

        for a in ["X", "Y", "Z"]:
            slider_layout = QHBoxLayout()
            slider_layout.addWidget(QLabel(f"{a} 轴"), 0)
            s = QSlider(Qt.Orientation.Horizontal)
            s.valueChanged.connect(self.schedule_update)
            slider_layout.addWidget(s, 1)
            ctrl_panel.addLayout(slider_layout)
            self.sliders[a] = s

        btn_home = QPushButton("<- 返回首页")
        btn_home.clicked.connect(self.show_home)
        ctrl_panel.addWidget(btn_home)

        workspace_layout.addWidget(view_container, 1)
        workspace_layout.addWidget(ctrl_container, 0)
        main_layout.addLayout(workspace_layout, 1)
        return viewer

    # ==================== 患者记录管理 ====================
    def _get_patients_db_path(self):
        return os.path.join(os.path.abspath("."), "patient_records.json")

    def _load_patients(self):
        db_path = self._get_patients_db_path()
        if os.path.exists(db_path):
            try:
                with open(db_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save_patients(self, patients):
        db_path = self._get_patients_db_path()
        with open(db_path, "w", encoding="utf-8") as f:
            json.dump(patients, f, ensure_ascii=False, indent=2)

    def _refresh_patient_tree(self):
        self.patient_tree.clear()
        patients = self._load_patients()
        for i, p in enumerate(patients):
            item = QTreeWidgetItem()
            item.setText(0, str(i + 1))
            item.setText(1, p.get("name", ""))
            item.setText(2, p.get("gender", ""))
            item.setText(3, p.get("age", ""))
            item.setText(4, p.get("head_circumference", ""))
            item.setText(5, p.get("disease_type", ""))
            item.setText(6, p.get("notes", ""))
            item.setText(7, p.get("created_at", ""))
            item.setData(0, Qt.ItemDataRole.UserRole, i)
            self.patient_tree.addTopLevelItem(item)

    def on_add_patient_clicked(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("新建患者记录")
        dlg.setMinimumWidth(480)
        dlg.setStyleSheet("QDialog { background-color: #111318; } QLabel { color: #CDD6F4; font-size: 12px; }")
        layout = QVBoxLayout(dlg)
        layout.setSpacing(10)

        title_label = QLabel("<b style='color:#A580FF; font-size:14px;'>患者信息录入</b>")
        layout.addWidget(title_label)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        line_style = "background-color: #181B22; color: white; border:1px solid #2A2E38; border-radius:4px; padding:4px;"

        ed_name = QLineEdit()
        ed_name.setPlaceholderText("请输入患者姓名")
        ed_name.setStyleSheet(line_style)
        form.addRow("患者姓名:", ed_name)

        ed_age = QLineEdit()
        ed_age.setPlaceholderText("例如 45")
        ed_age.setStyleSheet(line_style)
        form.addRow("患者年龄:", ed_age)

        ed_gender = QLineEdit()
        ed_gender.setPlaceholderText("男 / 女")
        ed_gender.setStyleSheet(line_style)
        form.addRow("患者性别:", ed_gender)

        ed_head = QLineEdit()
        ed_head.setPlaceholderText("例如 58 cm")
        ed_head.setStyleSheet(line_style)
        form.addRow("患者头围:", ed_head)

        ed_disease = QLineEdit()
        ed_disease.setPlaceholderText("例如 帕金森病")
        ed_disease.setStyleSheet(line_style)
        form.addRow("患病类型:", ed_disease)

        ed_notes = QLineEdit()
        ed_notes.setPlaceholderText("备注信息（可选）")
        ed_notes.setStyleSheet(line_style)
        form.addRow("备注:", ed_notes)

        layout.addLayout(form)

        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btn_box.button(QDialogButtonBox.StandardButton.Ok).setText("保存记录")
        btn_box.button(QDialogButtonBox.StandardButton.Ok).setStyleSheet("background-color: #1E66F5; color: white; padding: 6px 16px; border-radius: 6px; font-weight: bold;")
        btn_box.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        layout.addWidget(btn_box)

        btn_box.accepted.connect(dlg.accept)
        btn_box.rejected.connect(dlg.reject)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        name = ed_name.text().strip()
        if not name:
            QMessageBox.warning(self, "缺少信息", "患者姓名不能为空。")
            return

        import time
        new_patient = {
            "name": name,
            "age": ed_age.text().strip(),
            "gender": ed_gender.text().strip(),
            "head_circumference": ed_head.text().strip(),
            "disease_type": ed_disease.text().strip(),
            "notes": ed_notes.text().strip(),
            "created_at": time.strftime("%Y-%m-%d %H:%M"),
        }

        patients = self._load_patients()
        patients.append(new_patient)
        self._save_patients(patients)
        self._refresh_patient_tree()
        QMessageBox.information(self, "记录已保存", f"患者「{name}」的信息已成功录入。")

    def on_delete_patient_clicked(self):
        selected = self.patient_tree.currentItem()
        if selected is None:
            QMessageBox.warning(self, "未选择", "请先在列表中选择要删除的患者记录。")
            return
        idx = selected.data(0, Qt.ItemDataRole.UserRole)
        patients = self._load_patients()
        if idx is None or idx >= len(patients):
            return
        name = patients[idx].get("name", "未知")
        reply = QMessageBox.question(self, "确认删除", f"确定要删除患者「{name}」的记录吗？\n此操作不可撤销。",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            del patients[idx]
            self._save_patients(patients)
            self._refresh_patient_tree()

    def create_patient_records_page(self):
        page = QWidget()
        main_layout = QVBoxLayout(page)
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(12)

        header_layout = QHBoxLayout()
        title = QLabel(" 患者记录管理")
        title.setStyleSheet("color: white; font-size: 22px; font-weight: bold;")
        header_layout.addWidget(title)
        header_layout.addStretch()

        btn_add = QPushButton("+ 新建患者记录")
        btn_add.setObjectName("primaryBtn")
        btn_add.setMinimumHeight(38)
        btn_add.setStyleSheet("font-size: 14px; padding: 6px 20px;")
        btn_add.clicked.connect(self.on_add_patient_clicked)
        header_layout.addWidget(btn_add)

        btn_delete = QPushButton("删除选中记录")
        btn_delete.setStyleSheet("font-size: 14px; padding: 6px 20px; background-color: #B71C1C; color: white; border: none; border-radius: 6px;")
        btn_delete.setMinimumHeight(38)
        btn_delete.clicked.connect(self.on_delete_patient_clicked)
        header_layout.addWidget(btn_delete)

        btn_back = QPushButton("<- 返回首页")
        btn_back.setMinimumHeight(38)
        btn_back.clicked.connect(self.show_home)
        header_layout.addWidget(btn_back)

        main_layout.addLayout(header_layout)

        self.patient_tree = QTreeWidget()
        self.patient_tree.setColumnCount(8)
        self.patient_tree.setHeaderLabels(["#", "姓名", "性别", "年龄", "头围", "患病类型", "备注", "录入时间"])
        self.patient_tree.setStyleSheet("""
            QTreeWidget {
                background-color: #181B22;
                border: 1px solid #2A2E38;
                border-radius: 8px;
                color: #CDD6F4;
                font-size: 13px;
            }
            QTreeWidget::item {
                padding: 6px 4px;
                border-bottom: 1px solid #2A2E38;
            }
            QTreeWidget::item:selected {
                background-color: #1E66F5;
            }
            QHeaderView::section {
                background-color: #222530;
                color: #A580FF;
                padding: 8px 4px;
                border: none;
                border-bottom: 2px solid #2A2E38;
                font-weight: bold;
                font-size: 12px;
            }
        """)
        self.patient_tree.setAlternatingRowColors(True)
        header_view = self.patient_tree.header()
        header_view.setStretchLastSection(True)
        header_view.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header_view.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header_view.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header_view.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header_view.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header_view.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        header_view.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        header_view.setSectionResizeMode(7, QHeaderView.ResizeMode.ResizeToContents)

        main_layout.addWidget(self.patient_tree, 1)

        info_label = QLabel(f"数据存储位置: {self._get_patients_db_path()}")
        info_label.setStyleSheet("color: #7A8499; font-size: 10px;")
        main_layout.addWidget(info_label)

        self._refresh_patient_tree()
        return page

    # ==================== 靶点定位 ====================
    # MNI152 标准空间靶点坐标 (mm)
    TARGET_COORDS_MNI = {
        "lGPi":         (-20, -7, -6),
        "rGPi":         (20, -7, -6),
        "STN":          (-12, -15, -7),
        "Vim":          (13, -16, 0),
        "lM1":          (-37, -21, 58),
        "rM1":          (38, -22, 58),
        "M1":           (-37, -21, 58),
        "SMA_normal":   (0, 18, 58),
        "SMA_pre":      (0, -6, 60),
        "Right_putamen":(24, 1, -1),
        "Right_pallidum": (18, 2, -4),
        "对照_Left_putamen": (-24, 1, -1),

        "Left_DLPFC":     (-38, 44, 26),
        "sgAcc":          (0, 25, -10),
        "Left_NAc":       (-9, 10, -6),
        "Right_Amygdala": (24, -4, -18),
        "rVS":            (10, 8, -8),
        "对照_Left_Amygdala": (-24, -4, -18),
        "对照_Right_NAc":     (9, 10, -6),

        "Left_Hippocampus":  (-26, -16, -18),
        "Right_Hippocampus": (26, -16, -18),
        "对照_Right_Hippocampus": (26, -16, -18),

        "Left_Amygdala":  (-24, -4, -18),

        "Sphere_dmPFC":       (0, 50, 30),
        "WM_hypointensities": (0, -10, 22),
        "Right_caudate":      (12, 12, 8),
        "Right_VentralDC":    (12, -14, -4),
        "Right_vessel":       (10, -8, 4),
    }

    def on_goto_target_clicked(self):
        """一键定位至当前已选靶点的 MNI152 标准坐标"""
        if self.engine.data is None:
            QMessageBox.warning(self, "缺少底图", "请先加载底图再定位靶点。")
            return
        if self.target_combo.currentIndex() <= 0:
            QMessageBox.warning(self, "未选择靶点", "请先在【刺激靶点与图谱配置】中选择一个靶点。")
            return
        target = self.target_combo.currentText().strip()
        coords = self.TARGET_COORDS_MNI.get(target)
        if coords is None:
            QMessageBox.information(self, "坐标未知",
                f"靶点「{target}」的 MNI152 坐标暂未收录。\n\n"
                "已收录的靶点包括：lGPi、rGPi、丘脑底核、lM1、rM1、"
                "SMA_normal、SMA_pre、Left_DLPFC、sgAcc、rVS、"
                "左/右侧杏仁核、左/右海马体 等。")
            return
        mnix, mniy, mniz = coords
        try:
            vx, vy, vz = self.engine.mni_to_voxel(mnix, mniy, mniz)
            sh = self.engine.data.shape
            # 边界检查
            vx = int(np.clip(vx, 0, sh[0] - 1))
            vy = int(np.clip(vy, 0, sh[1] - 1))
            vz = int(np.clip(vz, 0, sh[2] - 1))
            # 更新滑块
            for s in self.sliders.values():
                s.blockSignals(True)
            self.sliders["X"].setValue(vx)
            self.sliders["Y"].setValue(vy)
            self.sliders["Z"].setValue(vz)
            for s in self.sliders.values():
                s.blockSignals(False)
            self.schedule_update()
            QMessageBox.information(self, "定位完成",
                f"已定位至靶点「{target}」\n"
                f"MNI152 坐标：({mnix}, {mniy}, {mniz}) mm\n"
                f"体素坐标：({vx}, {vy}, {vz})")
        except Exception as e:
            QMessageBox.warning(self, "定位失败", f"无法计算靶点坐标：{e}")

    # ==================== 个体精准模式按钮回调 ====================
    def on_individual_mode_clicked(self):
        visible = self.base_map_section.isVisible()
        self.base_map_section.setVisible(not visible)
        if not visible:
            self.btn_individual_mode.setText("个体精准模式 ^")
        else:
            self.btn_individual_mode.setText("个体精准模式")

    # ==================== 仿真报告导出 ====================
    def _find_electrode_map(self, disease, target):
        """返回当前疾病+靶点对应的电极图路径，或 None。"""
        ext_list = [".png", ".jpg", ".jpeg"]
        base_dir = os.path.join(os.path.abspath("."), "assets", "electrode_maps")
        if getattr(sys, 'frozen', False):
            base_dir = os.path.join(sys._MEIPASS, "assets", "electrode_maps")
        if not disease or not target or not os.path.isdir(base_dir):
            return None
        disease_dir = os.path.join(base_dir, disease)
        # 1. 精确匹配
        for ext in ext_list:
            candidate = os.path.join(disease_dir, target + ext)
            if os.path.exists(candidate):
                return candidate
        # 2. 模糊匹配
        if os.path.isdir(disease_dir):
            for fname in sorted(os.listdir(disease_dir)):
                if target.lower() in fname.lower():
                    full = os.path.join(disease_dir, fname)
                    if os.path.isfile(full):
                        return full
        # 3. default 兜底
        if os.path.isdir(disease_dir):
            for fname in sorted(os.listdir(disease_dir)):
                fnl = fname.lower()
                if fnl.startswith("default") or fnl.startswith("通用"):
                    return os.path.join(disease_dir, fname)
        return None

    def _fill_report_from_patient(self, combo_index):
        """当下拉选择已有患者记录时，自动填充表单字段"""
        if combo_index <= 0 or not hasattr(self, "_report_eds"):
            return
        patients = self._load_patients()
        patient_idx = combo_index - 1  # index 0 = "手动填写"
        if patient_idx < 0 or patient_idx >= len(patients):
            return
        p = patients[patient_idx]
        self._report_eds["name"].setText(p.get("name", ""))
        self._report_eds["age"].setText(p.get("age", ""))
        self._report_eds["gender"].setText(p.get("gender", ""))
        self._report_eds["head"].setText(p.get("head_circumference", ""))
        # 疾病类型由前端靶点配置锁定，不从患者记录覆盖

    def on_export_report_clicked(self):
        if self.engine.data is None:
            QMessageBox.warning(self, "缺少底图", "请先加载解剖底图再导出报告。")
            return

        disease = self.disease_combo.currentText() if self.disease_combo.currentIndex() > 0 else ""
        target = self.target_combo.currentText() if self.target_combo.isEnabled() and self.target_combo.currentIndex() > 0 else ""

        # ── 弹窗收集信息 ──
        dlg = QDialog(self)
        dlg.setWindowTitle("仿真报告信息填写")
        dlg.setMinimumWidth(520)
        dlg.setStyleSheet("QDialog { background-color: #111318; } QLabel { color: #CDD6F4; font-size: 12px; }")
        layout = QVBoxLayout(dlg)
        layout.setSpacing(10)

        line_style = "background-color: #181B22; color: white; border:1px solid #2A2E38; border-radius:4px; padding:4px;"

        # ── 关联已有患者记录 ──
        patients = self._load_patients()
        patient_label = QLabel("<b style='color:#A580FF; font-size:13px;'>患者基本信息</b>")
        layout.addWidget(patient_label)

        link_layout = QHBoxLayout()
        self.report_patient_combo = QComboBox()
        self.report_patient_combo.addItem("手动填写（不关联已有记录）")
        for p in patients:
            self.report_patient_combo.addItem(f"{p.get('name','')} | {p.get('disease_type','')} | {p.get('created_at','')}")
        self.report_patient_combo.currentIndexChanged.connect(lambda idx: self._fill_report_from_patient(idx))
        link_layout.addWidget(QLabel("关联记录:"), 0)
        link_layout.addWidget(self.report_patient_combo, 1)
        layout.addLayout(link_layout)

        form_patient = QFormLayout()
        form_patient.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        ed_name = QLineEdit()
        ed_name.setPlaceholderText("请输入患者姓名")
        ed_name.setStyleSheet(line_style)
        form_patient.addRow("患者姓名:", ed_name)

        ed_age = QLineEdit()
        ed_age.setPlaceholderText("例如 45")
        ed_age.setStyleSheet(line_style)
        form_patient.addRow("患者年龄:", ed_age)

        ed_gender = QLineEdit()
        ed_gender.setPlaceholderText("男 / 女")
        ed_gender.setStyleSheet(line_style)
        form_patient.addRow("患者性别:", ed_gender)

        ed_head = QLineEdit()
        ed_head.setPlaceholderText("例如 58 cm")
        ed_head.setStyleSheet(line_style)
        form_patient.addRow("患者头围:", ed_head)

        # 疾病类型自动从 disease_combo 填充
        ed_disease_type = QLineEdit()
        ed_disease_type.setText(disease)
        ed_disease_type.setStyleSheet(line_style + " color: #A0A8C0;")
        form_patient.addRow("患病类型:", ed_disease_type)

        layout.addLayout(form_patient)

        # 存储引用供 combo 回调使用
        self._report_eds = {
            "name": ed_name, "age": ed_age, "gender": ed_gender,
            "head": ed_head, "disease": ed_disease_type,
        }

        layout.addWidget(self._make_divider())

        # 仿真设置
        sim_label = QLabel("<b style='color:#A580FF; font-size:13px;'>仿真设置</b>")
        layout.addWidget(sim_label)
        form_sim = QFormLayout()
        form_sim.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        ed_target = QLineEdit()
        ed_target.setText(target)
        ed_target.setReadOnly(True)
        ed_target.setStyleSheet(line_style + " color: #A0A8C0;")
        form_sim.addRow("靶区:", ed_target)

        ed_current = QLineEdit()
        ed_current.setPlaceholderText("例如 2 mA")
        ed_current.setStyleSheet(line_style)
        form_sim.addRow("刺激电流:", ed_current)

        layout.addLayout(form_sim)

        # 电极位置图（自动查找）
        electrode_path = self._find_electrode_map(disease, target)
        elec_label = QLabel(f"电极位置图: {'已找到' if electrode_path else '未找到（可在 assets/electrode_maps/ 下放置）'}")
        elec_label.setStyleSheet("color: #00FF00; font-size: 10px;" if electrode_path else "color: #FF6600; font-size: 10px;")
        layout.addWidget(elec_label)

        # 确认 / 取消
        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btn_box.button(QDialogButtonBox.StandardButton.Ok).setText("确认生成报告")
        btn_box.button(QDialogButtonBox.StandardButton.Ok).setStyleSheet("background-color: #1E66F5; color: white; padding: 6px 16px; border-radius: 6px; font-weight: bold;")
        btn_box.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        layout.addWidget(btn_box)

        btn_box.accepted.connect(dlg.accept)
        btn_box.rejected.connect(dlg.reject)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        # ── 收集数据，生成报告 ──
        patient_info = {
            "name": ed_name.text().strip(),
            "age": ed_age.text().strip(),
            "gender": ed_gender.text().strip(),
            "head_circumference": ed_head.text().strip(),
            "disease_type": ed_disease_type.text().strip(),
            "current": ed_current.text().strip(),
            "electrode_map_path": electrode_path,
        }

        try:
            from tools.generate_pdf import generate_report
            success, info = generate_report(
                self.engine,
                self.current_base_path or "",
                disease,
                target,
                canvases=self.canvases,
                patient_info=patient_info,
            )
            if success:
                QMessageBox.information(self, "报告生成完毕", f"仿真报告已保存至:\n{info}")
            else:
                QMessageBox.warning(self, "报告生成失败", info)
        except ImportError:
            QMessageBox.warning(self, "缺少依赖", "请安装 reportlab:\npip install reportlab")
        except Exception as exc:
            QMessageBox.warning(self, "报告生成出错", str(exc))

    def _make_divider(self):
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setStyleSheet("background-color: #2A2E38; max-height: 1px;")
        return line

    # ==================== 电极图 ====================
    def on_electrode_map_clicked(self):
        """10-10 电极位置图：按当前选中的靶点查找对应电极图"""
        # 获取当前选中的靶点名称（例如 lGPi、Left_DLPFC 等）
        target = ""
        disease = ""
        if self.target_combo.isEnabled() and self.target_combo.currentIndex() > 0:
            target = self.target_combo.currentText()
        if self.disease_combo.currentIndex() > 0:
            disease = self.disease_combo.currentText()

        # 构建查找路径：assets/electrode_maps/{疾病名}/{靶点名}.png/.jpg
        ext_list = [".png", ".jpg", ".jpeg"]
        base_dir = os.path.join(os.path.abspath("."), "assets", "electrode_maps")
        if getattr(sys, 'frozen', False):
            base_dir = os.path.join(sys._MEIPASS, "assets", "electrode_maps")

        found = None
        label_found = ""

        if target and disease and os.path.isdir(base_dir):
            disease_dir = os.path.join(base_dir, disease)

            # 1. 优先按疾病子目录 + 靶点精确匹配
            for ext in ext_list:
                candidate = os.path.join(disease_dir, target + ext)
                if os.path.exists(candidate):
                    found = candidate
                    label_found = f"{disease} / {target}"
                    break

            # 2. 靶点模糊匹配（文件名包含靶点关键词）
            if not found and os.path.isdir(disease_dir):
                for fname in sorted(os.listdir(disease_dir)):
                    if target.lower() in fname.lower():
                        full = os.path.join(disease_dir, fname)
                        if os.path.isfile(full):
                            found = full
                            label_found = f"{disease} / {target}"
                            break

            # 3. 回退到疾病目录下的 default / 通用
            if not found and os.path.isdir(disease_dir):
                for fname in sorted(os.listdir(disease_dir)):
                    fnl = fname.lower()
                    if fnl.startswith("default") or fnl.startswith("通用"):
                        found = os.path.join(disease_dir, fname)
                        label_found = f"{disease} (通用)"
                        break

        # 4. 全局兜底
        if not found and os.path.isdir(base_dir):
            for fname in sorted(os.listdir(base_dir)):
                if fname.lower().startswith("default") or fname.lower().startswith("通用"):
                    found = os.path.join(base_dir, fname)
                    label_found = "通用"
                    break

        if found:
            from PyQt6.QtGui import QPixmap
            from PyQt6.QtWidgets import QDialog, QVBoxLayout as QVBox

            title_text = f"10-10 电极位置图 — {label_found}" if label_found else "10-10 电极位置图"
            dlg = QDialog(self)
            dlg.setWindowTitle(title_text)
            dlg.resize(700, 500)
            layout = QVBox(dlg)

            lbl_img = QLabel()
            pix = QPixmap(found)
            if not pix.isNull() and pix.width() > 0:
                scaled = pix.scaled(680, 460, Qt.AspectRatioMode.KeepAspectRatio,
                                   Qt.TransformationMode.SmoothTransformation)
                lbl_img.setPixmap(scaled)
                lbl_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
            else:
                lbl_img.setText(f"图片加载失败: {found}")
            layout.addWidget(lbl_img, 1)

            info = QLabel(f"文件: {found}")
            info.setStyleSheet("color: #7A8499; font-size: 10px;")
            info.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(info)

            dlg.exec()
        else:
            # 提示用户存放位置
            example_lines = []
            if disease:
                example_dir = os.path.join(base_dir, disease)
                example_lines.append(f"疾病目录: {example_dir}\\")
                example_lines.append(f"  • {target}.png (当前靶点)" if target else "  • {靶点名}.png")
                example_lines.append("  • default.png (该疾病通用)")
            else:
                example_lines.append(f"电极图目录: {base_dir}\\")
                example_lines.append("  {疾病}\\{靶点}.png")
            QMessageBox.information(self, "10-10 电极图",
                f"未找到电极图文件。\n\n"
                + "\n".join(example_lines) + "\n\n"
                f"当前疾病: {'「' + disease + '」' if disease else '无'}\n"
                f"当前靶点: {'「' + target + '」' if target else '无（请先选择靶点）'}\n\n"
                "支持格式: PNG / JPG / JPEG")

    # ==================== 底图加载 ====================
    def on_standard_brain_clicked(self):
        """标准高效模式：加载标准人脑模板"""
        self.steps_frame.setVisible(False)
        self.btn_std_mode.setEnabled(False)
        self.btn_std_mode.setText("正在加载标准模板…")
        QApplication.processEvents()

        std_file = os.path.join(get_mri_data_path(), "T1.nii")
        if not os.path.exists(std_file):
            self.btn_std_mode.setEnabled(True)
            self.btn_std_mode.setText("标准高效模式")
            QMessageBox.warning(self, "找不到标准脑",
                "无法检测到标准脑影像底图(T1.nii)。\n"
                f"已检查路径: {get_mri_data_path()}")
            return

        success, info = self.engine.load_image(std_file)
        if not success:
            self.btn_std_mode.setEnabled(True)
            self.btn_std_mode.setText("标准高效模式")
            QMessageBox.warning(self, "加载失败", f"无法加载标准模板:\n{info}")
            return

        self.current_base_path = std_file
        self.engine.overlays.clear()
        self.layer_list.clear()
        self.label_list.clear()
        self.mask_panel.setVisible(False)
        sh = self.engine.data.shape
        for i, a in enumerate(["X", "Y", "Z"]):
            self.sliders[a].setRange(0, sh[i] - 1)
            self.sliders[a].setValue(sh[i] // 2)
        self.schedule_update()

        self.btn_std_mode.setEnabled(True)
        self.btn_std_mode.setText("标准高效模式")

        spacing = [np.linalg.norm(self.engine.affine[0:3, i]) for i in range(3)]
        QMessageBox.information(self, "标准高效模式",
            f"标准人脑模板已成功加载。<br><br>"
            f"<b>模板信息：</b><br>"
            f"Shape: {sh[0]}×{sh[1]}×{sh[2]}<br>"
            f"Spacing: {spacing[0]:.2f}×{spacing[1]:.2f}×{spacing[2]:.2f} mm<br><br>"
            f"您可按需配置靶点与图谱。")

    # ==================== 其他辅助方法 ====================
    def refresh_layer_list_ui(self, schedule=True):
        self.layer_list.blockSignals(True)
        self.layer_list.clear()
        for ov in self.engine.overlays:
            name = ("[Atlas] " if ov.get("is_atlas") else "") + ov["name"]
            item = QListWidgetItem(name)
            item.setCheckState(Qt.CheckState.Checked)
            self.layer_list.addItem(item)
        if self.layer_list.count() > 0:
            self.layer_list.setCurrentRow(self.layer_list.count() - 1)
        self.layer_list.blockSignals(False)
        if schedule:
            self.schedule_update()

    def scan_local_mri_files(self):
        target_path = get_mri_data_path()
        self.mri_combo.blockSignals(True)
        self.mri_combo.clear()
        self.mri_combo.addItem("快速选择本地目录 MRI...")
        if os.path.exists(target_path):
            for file in sorted(os.listdir(target_path)):
                if file.endswith(".nii") or file.endswith(".nii.gz"):
                    self.mri_combo.addItem(file)
        self.mri_combo.blockSignals(False)

    def on_local_mri_changed(self, index):
        if index <= 0:
            return
        filename = self.mri_combo.currentText()
        self.execute_base_mri_load(os.path.join(get_mri_data_path(), filename))

    def on_browse_mri_clicked(self):
        """浏览按钮：打开文件对话框选择任意 NIfTI 文件加载，并记忆路径"""
        start_dir = self._last_browse_dir if self._last_browse_dir else get_mri_data_path()
        p, _ = QFileDialog.getOpenFileName(self, "选择 NIfTI 底图文件", start_dir, "医学影像 (*.nii *.nii.gz)")
        if p:
            self._last_browse_dir = os.path.dirname(p)
            self.execute_base_mri_load(p)

    def execute_base_mri_load(self, filepath):
        self.engine.overlays.clear()
        self.layer_list.clear()
        self.label_list.clear()
        self.mask_panel.setVisible(False)
        success, info = self.engine.load_image(filepath)
        if not success:
            QMessageBox.warning(self, "导入失败", f"错误解析: {info}")
            return
        self.current_base_path = filepath
        self._last_browse_dir = os.path.dirname(os.path.abspath(filepath))
        sh = self.engine.data.shape
        for i, a in enumerate(["X", "Y", "Z"]):
            self.sliders[a].setRange(0, sh[i] - 1)
            self.sliders[a].setValue(sh[i] // 2)
        self.schedule_update()

    def on_browse_overlay_clicked(self):
        p, _ = QFileDialog.getOpenFileName(self, "导入彩色追加层", get_mri_data_path(), "医学影像 (*.nii *.nii.gz)")
        if p:
            self.execute_overlay_mri_load(p)

    def execute_overlay_mri_load(self, filepath):
        if self.engine.data is None:
            QMessageBox.warning(self, "无法叠加", "无基础底图，请先载入背景。")
            return
        success, info = self.engine.load_overlay(filepath)
        if not success:
            QMessageBox.warning(self, "叠加失败", f"错误: {info}")
            return
        self.refresh_layer_list_ui()

    def refresh_disease_list(self):
        self.disease_combo.blockSignals(True)
        self.disease_combo.clear()
        self.disease_combo.addItem("选择临床疾病...")
        sample_dir = get_sample_data_path()
        if os.path.exists(sample_dir):
            for item in sorted(os.listdir(sample_dir)):
                if os.path.isdir(os.path.join(sample_dir, item)):
                    self.disease_combo.addItem(item)
        self.disease_combo.blockSignals(False)

    def on_disease_changed(self, index):
        self.target_combo.blockSignals(True)
        self.target_combo.clear()
        if index <= 0:
            self.target_combo.addItem("选择干预靶点...")
            self.target_combo.setEnabled(False)
            self.target_combo.blockSignals(False)
            return
        disease = self.disease_combo.currentText()
        disease_path = os.path.join(get_sample_data_path(), disease)
        self.target_combo.addItem("选择干预靶点...")
        if os.path.exists(disease_path):
            for fname in sorted(os.listdir(disease_path)):
                if os.path.isdir(os.path.join(disease_path, fname)):
                    self.target_combo.addItem(fname)
        self.target_combo.setEnabled(True)
        self.target_combo.blockSignals(False)

    # 图层名称转换映射
    @staticmethod
    def _friendly_layer_name(filename):
        """将内部文件名转换为人类可读的图层名称"""
        name = filename
        for ext in ['.nii.gz', '.nii']:
            if name.endswith(ext):
                name = name[:-len(ext)]
                break
        # 替换常见模式
        if name.startswith('grey_flex_mean'):
            name = name.replace('grey_flex_mean', '灰质电场', 1)
        elif name.startswith('white_flex_mean'):
            name = name.replace('white_flex_mean', '白质电场', 1)
        elif name.startswith('flex_mean'):
            name = name.replace('flex_mean', '全组织电场', 1)
        # 去掉内部标记但保留靶点信息
        name = name.replace('_MNI', ' (MNI)')
        name = name.replace('_subject', ' (个体)')
        return name

    def on_target_changed(self, index):
        if index <= 0:
            return
        disease = self.disease_combo.currentText()
        target = self.target_combo.currentText()
        target_dir = os.path.join(get_sample_data_path(), disease, target)
        if not os.path.exists(target_dir):
            return
        all_files = [f for f in sorted(os.listdir(target_dir)) if f.endswith(".nii") or f.endswith(".nii.gz")]
        if not all_files:
            return
        if self.engine.data is None:
            QMessageBox.warning(self, "缺少底图", "请先加载底图层！")
            self.target_combo.setCurrentIndex(0)
            return

        tissue_files = [
            f for f in all_files
            if f.lower().startswith(("grey_flex_mean", "white_flex_mean"))
        ]
        if not tissue_files:
            tissue_files = [f for f in all_files if not f.lower().startswith("flex_mean")]
        if not tissue_files:
            tissue_files = all_files

        same_space_files = [
            f for f in tissue_files
            if self.engine.file_matches_base_space(os.path.join(target_dir, f))
        ]
        if not same_space_files:
            self.engine.overlays.clear()
            self.layer_list.clear()
            self.label_list.clear()
            self.mask_panel.setVisible(False)
            QMessageBox.warning(
                self,
                "需要先完成仿真",
                "如果要使用外部个体 T1，"
                "需要先进行对应靶点仿真。"
            )
            return

        filtered_files = same_space_files

        self.engine.overlays.clear()
        self.layer_list.clear()
        self.label_list.clear()
        self.mask_panel.setVisible(False)
        for fname in filtered_files:
            success, info = self.engine.load_overlay(os.path.join(target_dir, fname))
            if success:
                self.engine.overlays[-1]["name"] = self._friendly_layer_name(fname)
        self.refresh_layer_list_ui()

    def _base_matches_builtin_t1(self):
        std_file = os.path.join(get_mri_data_path(), "T1.nii")
        return os.path.exists(std_file) and self.engine.file_matches_base_space(std_file)

    def on_load_aal_clicked(self):
        aal_dir = get_sample_data_path()
        aal_nii = os.path.join(aal_dir, "aal.nii")
        if not os.path.exists(aal_nii):
            aal_nii = os.path.join(aal_dir, "aal.nii.gz")
        if not os.path.exists(aal_nii) or self.engine.data is None:
            return
        if not self._base_matches_builtin_t1() and not self.engine.file_matches_base_space(aal_nii):
            QMessageBox.warning(
                self,
                "AAL 图谱需要仿真",
                "当前解剖底图不是内置标准模板，也不是 AAL 图谱本身所在空间。\n\n"
                "请先完成个体 T1 与模板/AAL 空间的仿真，再加载变换到个体空间后的 AAL 图谱。"
            )
            return
        success, info = self.engine.load_overlay(aal_nii)
        if not success:
            return
        idx = info
        self.engine.overlays[idx]["is_atlas"] = True
        aal_labels_txt = os.path.join(aal_dir, "aal_labels.txt")
        if os.path.exists(aal_labels_txt):
            self.engine.load_label_dict(idx, aal_labels_txt)
        self.refresh_layer_list_ui()

    def _safe_path_name(self, text):
        bad_chars = '<>:"/\\|?*'
        clean = "".join("_" if ch in bad_chars else ch for ch in text)
        return clean.strip().replace(" ", "_") or "unnamed"

    def on_register_current_target_clicked(self):
        if self.engine.data is None or not self.current_base_path:
            QMessageBox.warning(self, "缺少底图", "请先加载个体 T1。")
            return
        if self.target_combo.currentIndex() <= 0 or self.disease_combo.currentIndex() <= 0:
            QMessageBox.warning(self, "缺少靶点", "请先选择疾病和干预靶点。")
            return

        template_t1 = os.path.join(get_mri_data_path(), "T1.nii")
        if not os.path.exists(template_t1):
            QMessageBox.warning(self, "缺少模板", f"找不到内置模板:\n{template_t1}")
            return
        if self.engine.file_matches_base_space(template_t1):
            QMessageBox.information(self, "无需仿真", "当前底图已经是内置模板空间，可直接选择靶点加载仿真层。")
            return

        disease = self.disease_combo.currentText()
        target = self.target_combo.currentText()
        target_dir = os.path.join(get_sample_data_path(), disease, target)
        if not os.path.isdir(target_dir):
            QMessageBox.warning(self, "缺少靶点目录", f"找不到目录:\n{target_dir}")
            return

        all_files = [
            f for f in sorted(os.listdir(target_dir))
            if f.endswith(".nii") or f.endswith(".nii.gz")
        ]
        tissue_files = [
            f for f in all_files
            if f.lower().startswith(("grey_flex_mean", "white_flex_mean"))
        ]
        template_space_overlays = [
            os.path.join(target_dir, f)
            for f in tissue_files
            if NiiEngine.files_match_space(os.path.join(target_dir, f), template_t1)
        ]
        if not template_space_overlays:
            QMessageBox.warning(
                self,
                "无可仿真图层",
                "当前靶点目录中没有找到与内置模板 T1 同空间的灰/白质仿真层。"
            )
            return

        base_name = os.path.basename(self.current_base_path)
        for ext in (".nii.gz", ".nii"):
            if base_name.endswith(ext):
                base_name = base_name[:-len(ext)]
                break
        out_dir = os.path.join(
            os.path.abspath("."),
            "registered",
            self._safe_path_name(base_name),
            self._safe_path_name(disease),
            self._safe_path_name(target),
        )

        self.btn_register_target.setEnabled(False)
        self.registration_progress = QProgressDialog("正在进行非线性仿真，请稍候…", None, 0, 0, self)
        self.registration_progress.setWindowTitle("个体 T1 仿真")
        self.registration_progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.registration_progress.setCancelButton(None)
        self.registration_progress.show()

        self.registration_worker = RegistrationWorker(
            self.current_base_path,
            template_t1,
            template_space_overlays,
            out_dir,
            self,
        )
        self.registration_worker.finished.connect(self.on_registration_finished)
        self.registration_worker.start()

    def on_registration_finished(self, success, message, output_files):
        if hasattr(self, "registration_progress") and self.registration_progress:
            self.registration_progress.close()
        self.btn_register_target.setEnabled(True)
        if hasattr(self, "registration_worker") and self.registration_worker:
            self.registration_worker.deleteLater()
            self.registration_worker = None

        if not success:
            QMessageBox.warning(self, "仿真失败", message)
            return

        loaded = 0
        for path in output_files:
            ok, _ = self.engine.load_overlay(path)
            if ok:
                self.engine.overlays[-1]["name"] = self._friendly_layer_name(os.path.basename(path))
                loaded += 1
        self.timer_2d.stop()
        self.timer_3d.stop()
        self.refresh_layer_list_ui(schedule=False)
        self.update_2d_views()
        if getattr(self.canvas_3d, "available", False):
            self.canvas_3d.mark_rebuild_needed()
            self.update_3d_view()  # 立即安全重建，消耗 _need_full_rebuild 标志，避免后续用户操作触发全量重建崩溃
        QMessageBox.information(
            self,
            "仿真完成",
            f"{message}\n\n已自动加载 {loaded} 个图层。\n\n"
            "3D 视图已完成安全刷新，可正常操作。"
        )

    def schedule_update(self):
        if self.engine.data is None:
            return
        self.timer_2d.start(15)
        if not self._drag_active:
            self.timer_3d.start(500)

    def sync_cmap_selector(self, row):
        if row < 0 or row >= len(self.engine.overlays):
            return
        ov = self.engine.overlays[row]
        idx = self.cmap_combo.findText(ov["cmap"])
        if idx >= 0:
            self.cmap_combo.blockSignals(True)
            self.cmap_combo.setCurrentIndex(idx)
            self.cmap_combo.blockSignals(False)
        self.alpha_slider.blockSignals(True)
        self.alpha_slider.setValue(int(ov.get("alpha", 0.7) * 100))
        self.alpha_slider.blockSignals(False)
        if ov.get("is_atlas"):
            self.mask_panel.setVisible(True)
            self.label_list.blockSignals(True)
            self.label_list.clear()
            for lbl in ov["labels"]:
                if lbl == 0:
                    continue
                r_name = ov.get("label_dict", {}).get(lbl, f"Label {lbl}")
                item = QListWidgetItem(f"{lbl}: {r_name}")
                item.setData(Qt.ItemDataRole.UserRole, int(lbl))
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked if lbl in ov["active_labels"] else Qt.CheckState.Unchecked)
                self.label_list.addItem(item)
            self.label_list.blockSignals(False)
        else:
            self.mask_panel.setVisible(False)

    def on_cmap_changed(self):
        row = self.layer_list.currentRow()
        if row >= 0 and row < len(self.engine.overlays):
            self.engine.overlays[row]["cmap"] = self.cmap_combo.currentText()
            self.schedule_update()

    def on_alpha_changed(self):
        row = self.layer_list.currentRow()
        if row >= 0 and row < len(self.engine.overlays):
            self.engine.overlays[row]["alpha"] = self.alpha_slider.value() / 100.0
            self.schedule_update()

    def on_label_selection_changed(self):
        row = self.layer_list.currentRow()
        if row < 0 or row >= len(self.engine.overlays):
            return
        active = []
        for i in range(self.label_list.count()):
            it = self.label_list.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                active.append(it.data(Qt.ItemDataRole.UserRole))
        self.engine.overlays[row]["active_labels"] = active
        self.schedule_update()

    def _set_sliders_from_canvas(self, axis, x, y):
        if self.engine.data is None:
            return
        sh = self.engine.data.shape
        for s in self.sliders.values():
            s.blockSignals(True)
        if axis == "axial":
            self.sliders["X"].setValue(int(np.clip(x, 0, sh[0] - 1)))
            self.sliders["Y"].setValue(int(np.clip(y, 0, sh[1] - 1)))
        elif axis == "coronal":
            self.sliders["X"].setValue(int(np.clip(x, 0, sh[0] - 1)))
            self.sliders["Z"].setValue(int(np.clip(y, 0, sh[2] - 1)))
        elif axis == "sagittal":
            self.sliders["Y"].setValue(int(np.clip(x, 0, sh[1] - 1)))
            self.sliders["Z"].setValue(int(np.clip(y, 0, sh[2] - 1)))
        for s in self.sliders.values():
            s.blockSignals(False)

    def handle_3d_slider_change(self, x, y, z):
        """3D视图底部滑块拖动 -> 同步到侧边栏滑块并刷新2D"""
        if self.engine.data is None:
            return
        sh = self.engine.data.shape
        for s in self.sliders.values():
            s.blockSignals(True)
        self.sliders["X"].setValue(int(np.clip(x, 0, sh[0] - 1)))
        self.sliders["Y"].setValue(int(np.clip(y, 0, sh[1] - 1)))
        self.sliders["Z"].setValue(int(np.clip(z, 0, sh[2] - 1)))
        for s in self.sliders.values():
            s.blockSignals(False)
        self.update_2d_views()

    def handle_canvas_click(self, axis, x, y):
        self._set_sliders_from_canvas(axis, x, y)
        self.schedule_update()

    def handle_canvas_drag(self, axis, x, y):
        self._drag_active = True
        self._set_sliders_from_canvas(axis, x, y)
        self.schedule_update()

    def handle_canvas_drag_release(self, axis, x, y):
        self._drag_active = False
        self._set_sliders_from_canvas(axis, x, y)
        # 松开后刷新 3D
        if self.engine.data is not None:
            self.timer_3d.start(300)

    def get_render_params(self):
        x, y, z = [self.sliders[a].value() for a in ["X", "Y", "Z"]]
        atlas_vol, active_labels = None, None
        if self.chk_apply_mask.isChecked():
            for i in range(self.layer_list.count()):
                if i < len(self.engine.overlays) and self.layer_list.item(i).checkState() == Qt.CheckState.Checked:
                    ov = self.engine.overlays[i]
                    if ov.get("is_atlas") and len(ov["active_labels"]) > 0:
                        atlas_vol, active_labels = ov["data"], ov["active_labels"]
                        break
        active_ovs = []
        for i in range(self.layer_list.count()):
            if i < len(self.engine.overlays) and self.layer_list.item(i).checkState() == Qt.CheckState.Checked:
                ov = self.engine.overlays[i]
                d_range = ov["max"] - ov["min"]
                cur_min = ov["min"] if d_range == 0 else ov["min"] + (self.min_slider.value() / 100.0) * d_range
                active_ovs.append({
                    "data": ov["data"], "is_atlas": ov.get("is_atlas", False),
                    "active_labels": ov.get("active_labels", []), "cmap": ov["cmap"],
                    "alpha": ov.get("alpha", 0.7), "min": cur_min, "max": ov["max"],
                    "name": ov["name"]
                })
        return x, y, z, atlas_vol, active_labels, active_ovs

    def update_2d_views(self):
        if self.engine.data is None:
            return
        x, y, z, atlas_vol, active_labels, active_ovs = self.get_render_params()
        mni = self.engine.voxel_to_mni(x, y, z)
        self.coord_label.setText(f"MNI: ({mni[0]:.1f}, {mni[1]:.1f}, {mni[2]:.1f})")

        # ── 更新电场信息：每行一个图层 ──
        # 先清除旧的标签
        for lbl in self.val_labels:
            self.val_labels_layout.removeWidget(lbl)
            lbl.deleteLater()
        self.val_labels.clear()

        # 底图值
        base_lbl = QLabel(f"底图: {float(self.engine.data[x, y, z]):.4f}")
        base_lbl.setStyleSheet("color: #00FF00; font-family: Consolas; font-size: 10px;")
        self.val_labels_layout.addWidget(base_lbl)
        self.val_labels.append(base_lbl)

        # 每个叠加层一行：只显示灰质和白质电场
        for ov in active_ovs:
            sh = ov["data"].shape
            val = float(ov["data"][x, y, z]) if x < sh[0] and y < sh[1] and z < sh[2] else 0.0
            name = ov["name"]
            if "灰质" in name or "grey" in name.lower():
                color = "#FFD700"
                label = "灰质电场"
            elif "白质" in name or "white" in name.lower():
                color = "#00FFFF"
                label = "白质电场"
            else:
                continue  # 跳过其他叠加层
            lbl = QLabel(f"{label}: {val:.4f}")
            lbl.setStyleSheet(f"color: {color}; font-family: Consolas; font-size: 10px;")
            self.val_labels_layout.addWidget(lbl)
            self.val_labels.append(lbl)

        sx, sy, sz = (1.0, 1.0, 1.0) if self.engine.affine is None else [np.linalg.norm(self.engine.affine[0:3, i]) for i in range(3)]
        sag_ovs, cor_ovs, axi_ovs = [], [], []
        for ov in active_ovs:
            sag_slc, cor_slc, axi_slc = ov["data"][x, :, :], ov["data"][:, y, :], ov["data"][:, :, z]
            if ov["is_atlas"]:
                sag_slc = np.where(np.isin(sag_slc, ov["active_labels"]), sag_slc, 0)
                cor_slc = np.where(np.isin(cor_slc, ov["active_labels"]), cor_slc, 0)
                axi_slc = np.where(np.isin(axi_slc, ov["active_labels"]), axi_slc, 0)
            elif atlas_vol is not None:
                sag_slc = np.where(np.isin(atlas_vol[x, :, :], active_labels), sag_slc, np.nan)
                cor_slc = np.where(np.isin(atlas_vol[:, y, :], active_labels), cor_slc, np.nan)
                axi_slc = np.where(np.isin(atlas_vol[:, :, z], active_labels), axi_slc, np.nan)
            sag_ovs.append({**ov, "data": sag_slc})
            cor_ovs.append({**ov, "data": cor_slc})
            axi_ovs.append({**ov, "data": axi_slc})

        self.canvases["sagittal"].render_slice(self.engine.data[x, :, :], aspect=sz / sy, crosshair=(y, z), overlays=sag_ovs, vmin=self.engine.display_min, vmax=self.engine.display_max)
        self.canvases["coronal"].render_slice(self.engine.data[:, y, :], aspect=sz / sx, crosshair=(x, z), overlays=cor_ovs, vmin=self.engine.display_min, vmax=self.engine.display_max)
        self.canvases["axial"].render_slice(self.engine.data[:, :, z], aspect=sy / sx, crosshair=(x, y), overlays=axi_ovs, vmin=self.engine.display_min, vmax=self.engine.display_max)

    def update_3d_view(self):
        if self.engine.data is None:
            return
        x, y, z = [self.sliders[a].value() for a in ["X", "Y", "Z"]]
        try:
            # Stable mode: keep 3D to anatomical slices only. VTK overlay
            # meshes are the common native-crash point in the packaged exe.
            self.canvas_3d.render_3d(
                self.engine.data,
                x,
                y,
                z,
                overlays=None,
                vmin=self.engine.display_min,
                vmax=self.engine.display_max,
            )
        except Exception as exc:
            print(f"3D render skipped: {exc}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = ZjuMriViewer()
    ex.show()
    sys.exit(app.exec())
