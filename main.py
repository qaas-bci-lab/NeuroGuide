# main.py

import sys
import os

# CUDA checks run before Qt/VTK and never share their GPU context with inference.
if "--cuda-probe" in sys.argv:
    from core.cuda_compat import probe_main
    raise SystemExit(probe_main())

# Enter the isolated worker before importing Qt/VTK (also used by frozen builds).
if "--individual-worker" in sys.argv:
    from tools.individual_worker import main as worker_main
    raise SystemExit(worker_main(sys.argv[sys.argv.index("--individual-worker") + 1]))

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
    QTreeWidget, QTreeWidgetItem, QHeaderView, QSplitter, QScrollArea, QPlainTextEdit, QProgressBar, QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QEvent, QSize
from core.engine import NiiEngine
from ui.canvas import MriCanvas, Freeview3DWidget






class WrappingResultLabel(QLabel):
    """Reserve the actual wrapped height so scheme lines cannot be clipped."""
    def setText(self,text):
        super().setText(text)
        QTimer.singleShot(0,self.fit_text)

    def resizeEvent(self,event):
        super().resizeEvent(event)
        QTimer.singleShot(0,self.fit_text)

    def fit_text(self):
        required=max(self.fontMetrics().height(),self.heightForWidth(max(50,self.width())))
        if self.minimumHeight()!=required:self.setMinimumHeight(required)


class SidebarScrollArea(QScrollArea):
    """Fit the workflow to the viewport; scroll only when its minimum needs it.

    Qt's automatic sizing otherwise reserves every wrapped label's preferred
    height, pushing result controls below the viewport even when they fit.
    """
    def setWidget(self, widget):
        super().setWidget(widget)
        widget.installEventFilter(self)
        self.fit_contents()

    def fit_contents(self):
        widget = self.widget()
        if widget is None: return
        size = QSize(self.viewport().width(), max(self.viewport().height(), widget.minimumSizeHint().height()))
        if widget.size() != size: widget.resize(size)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit_contents()

    def eventFilter(self, watched, event):
        if watched is self.widget() and event.type() == QEvent.Type.LayoutRequest:
            QTimer.singleShot(0, self.fit_contents)
        return super().eventFilter(watched, event)


class ZjuMriViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.engine = NiiEngine()
        self.sliders = {}
        self.current_base_path = None
        self.individual_mode = True
        self.current_stimulation = None
        self._individual_result = None

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
        from ui.branding import application_icon
        self.setWindowIcon(application_icon())
        if QApplication.instance(): QApplication.instance().setWindowIcon(self.windowIcon())
        self.resize(1480, min(980, max(760, QApplication.primaryScreen().availableGeometry().height()-40)))

        from ui.theme import WINDOW_STYLE
        self.setStyleSheet(WINDOW_STYLE)

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.home_page = self.create_home_page()
        self.viewer_page = self.create_viewer_page()
        self.patient_page = self.create_patient_records_page()

        self.stack.addWidget(self.home_page)
        self.stack.addWidget(self.viewer_page)
        self.stack.addWidget(self.patient_page)
        self.on_individual_mode_clicked()
        self.stack.setCurrentIndex(0)

    def show_home(self, *_signal_args):
        dialog = getattr(self, "individual_dialog", None)
        if dialog:
            dialog.stop_target_selection()
        self.stack.setCurrentIndex(0)

    def create_home_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        from ui.branding import logo_label, WORDMARK
        self.home_logo = logo_label(176, page, 'homeLogo')
        layout.addWidget(self.home_logo, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addSpacing(4)
        title = QLabel(WORDMARK)
        title.setStyleSheet("font-size: 36px; font-weight: bold;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("个体电场建模与 TI / tES 方案规划")
        subtitle.setStyleSheet("color: #a8afbd; font-size: 14px; margin-bottom: 30px;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

        btn_viewer = QPushButton("新建刺激方案")
        btn_viewer.setObjectName("primaryBtn")
        btn_viewer.setFixedSize(320, 70)
        btn_viewer.setStyleSheet("font-size: 16px; border-radius: 10px;")
        btn_viewer.clicked.connect(self.on_individual_mode_clicked)
        layout.addWidget(btn_viewer, alignment=Qt.AlignmentFlag.AlignCenter)

        layout.addSpacing(15)

        btn_patients = QPushButton("患者记录")
        btn_patients.setObjectName("secondaryBtn")
        btn_patients.setFixedSize(320, 45)
        btn_patients.setStyleSheet("font-size: 14px; border-radius: 8px;")
        btn_patients.clicked.connect(lambda *_args: self.stack.setCurrentIndex(2))
        layout.addWidget(btn_patients, alignment=Qt.AlignmentFlag.AlignCenter)
        return page

    def add_ui_divider(self, parent_layout):
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setStyleSheet("background-color: #353943; max-height: 1px;")
        parent_layout.addWidget(line)


    def create_viewer_page(self):
        viewer = QWidget()
        main_layout = QVBoxLayout(viewer)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        header=QHBoxLayout(); header.setSpacing(8)
        from ui.branding import logo_label, WORDMARK
        self.workspace_logo=logo_label(36, viewer, 'workspaceLogo'); header.addWidget(self.workspace_logo)
        title=QLabel(WORDMARK); title.setObjectName("workspaceTitle"); header.addWidget(title)
        self.workspace_caption=QLabel("刺激方案规划"); header.addWidget(self.workspace_caption); header.addStretch()
        self.workspace_file=QLabel("选择 MRI，或使用内置 MRI"); header.addWidget(self.workspace_file)
        btn_home=QPushButton('返回首页'); btn_home.setFixedHeight(28)
        btn_home.setStyleSheet('padding: 3px 10px;'); btn_home.clicked.connect(self.show_home); header.addWidget(btn_home)
        main_layout.addLayout(header)

        workspace_layout = QHBoxLayout()
        workspace_layout.setSpacing(8)

        view_container = QFrame()
        view_container.setMinimumWidth(400)
        view_layout = QGridLayout(view_container)
        view_layout.setContentsMargins(0, 0, 0, 0)
        self.canvases = {
            "sagittal": MriCanvas("矢状面 / 侧面", "sagittal"),
            "coronal": MriCanvas("冠状面 / 正面", "coronal"),
            "axial": MriCanvas("轴状面 / 横切面", "axial")
        }
        self.canvas_3d = Freeview3DWidget()
        self.canvas_3d.slice_changed.connect(self.handle_3d_slider_change)
        for c in self.canvases.values():
            c.clicked.connect(self.handle_canvas_click)
            c.drag_moved.connect(self.handle_canvas_drag)
            c.drag_released.connect(self.handle_canvas_drag_release)
            c.slice_scrolled.connect(self.handle_canvas_scroll)
            c.setToolTip('鼠标滚轮切换当前平面的切片；点击或拖动定位十字光标。')

        self.view_rows=QSplitter(Qt.Orientation.Vertical);self.view_rows.setChildrenCollapsible(False)
        slice_rows=[]
        for _ in range(2):
            container=QWidget();row_layout=QHBoxLayout(container);row_layout.setContentsMargins(0,0,0,0);row_layout.setSpacing(8)
            slice_rows.append(row_layout);self.view_rows.addWidget(container)
        self.view_rows.setSizes([440,440]);view_layout.addWidget(self.view_rows,1,0,2,2)
        self.slice_coord_labels={}; self.slice_panels={}
        for plane,axis,row,column in (('sagittal','X',1,0),('coronal','Y',1,1),('axial','Z',2,0)):
            frame=QFrame(); frame.setObjectName('slicePanel')
            pane=QVBoxLayout(frame); pane.setContentsMargins(0,0,0,0); pane.setSpacing(2)
            toolbar=QHBoxLayout(); toolbar.setContentsMargins(7,2,7,2)
            label=QLabel(f'{axis}：—'); label.setMinimumWidth(140)
            self.slice_coord_labels[axis]=label; toolbar.addWidget(label)
            slider=QSlider(Qt.Orientation.Horizontal); slider.setRange(0,0); slider.setEnabled(False)
            slider.valueChanged.connect(self.schedule_update); toolbar.addWidget(slider,1)
            self.sliders[axis]=slider
            pane.addLayout(toolbar); pane.addWidget(self.canvases[plane],1)
            self.slice_panels[axis]=frame; slice_rows[row-1].addWidget(frame,1)
        plane_panel=QFrame(); plane_layout=QVBoxLayout(plane_panel)
        plane_layout.setContentsMargins(0,0,0,0); plane_layout.setSpacing(2)
        plane_toolbar=QHBoxLayout(); plane_toolbar.setContentsMargins(7,2,7,2)
        plane_toolbar.addWidget(QLabel("2.5D 切面")); plane_toolbar.addStretch()
        self.reset_3d_button=QPushButton("重置视角"); self.reset_3d_button.setFixedHeight(24)
        self.reset_3d_button.setToolTip("恢复沿当前扫描方向观察的相机视角；不改变切片、靶区或电场。")
        self.reset_3d_button.clicked.connect(self.canvas_3d.reset_view)
        plane_toolbar.addWidget(self.reset_3d_button)
        plane_layout.addLayout(plane_toolbar); plane_layout.addWidget(self.canvas_3d,1)
        slice_rows[1].addWidget(plane_panel,1)
        view_layout.setRowStretch(1,1); view_layout.setRowStretch(2,1)

        ctrl_container = QFrame()
        self.ctrl_container = ctrl_container
        ctrl_container.setObjectName("ctrlPanel")

        ctrl_panel = QVBoxLayout(ctrl_container)
        self.ctrl_layout=ctrl_panel
        ctrl_panel.setContentsMargins(12, 12, 12, 12)
        ctrl_panel.setSpacing(6)
        ctrl_panel.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.individual_host = QFrame()
        self.individual_host_layout = QVBoxLayout(self.individual_host)
        self.individual_host_layout.setContentsMargins(0, 0, 0, 0)
        ctrl_panel.addWidget(self.individual_host, 1)

        self.add_ui_divider(ctrl_panel)

        # 方案与结果
        sim_label = QLabel("<b style='color:#c4ccd8; font-size:13px;'>方案与结果</b>")
        sim_label.setMaximumHeight(24)
        ctrl_panel.addWidget(sim_label)

        self.current_panel=QFrame(); current_layout=QVBoxLayout(self.current_panel)
        current_layout.setContentsMargins(0,0,0,0)
        self.current_label=WrappingResultLabel("刺激电流：尚未合成")
        self.current_label.setWordWrap(True)
        self.current_label.setStyleSheet("color: #ffd38b; font-size: 12px;")
        current_layout.addWidget(self.current_label)
        from ui.result_text import FREQUENCY_NOTE
        self.frequency_note=WrappingResultLabel(FREQUENCY_NOTE);self.frequency_note.setWordWrap(True)
        self.frequency_note.setStyleSheet('color:#9ba5b6;font-size:11px;');current_layout.addWidget(self.frequency_note)
        ctrl_panel.addWidget(self.current_panel)

        sim_btn_layout = QHBoxLayout()
        sim_btn_layout.setSpacing(6)

        btn_report = QPushButton("导出仿真报告")
        btn_report.setObjectName("secondaryBtn")
        btn_report.setMinimumHeight(32)
        btn_report.setToolTip("将当前可视化状态导出为 PDF 仿真报告")
        btn_report.clicked.connect(self.on_export_report_clicked)
        sim_btn_layout.addWidget(btn_report, 1)

        btn_electrode = QPushButton("查看电极")
        self.btn_electrode = btn_electrode
        btn_electrode.setMinimumHeight(32)
        btn_electrode.setToolTip("在 10-10 图上查看当前模型的电极位置和电流方案")
        btn_electrode.clicked.connect(self.show_individual_electrodes)
        sim_btn_layout.addWidget(btn_electrode, 1)

        ctrl_panel.addLayout(sim_btn_layout)

        self.add_ui_divider(ctrl_panel)

        # 公共控制区
        ctrl_panel.addWidget(QLabel("显示内容"))
        self.layer_list = QListWidget()
        self.layer_list.setFixedHeight(42)
        self.layer_list.currentRowChanged.connect(self.sync_cmap_selector)
        self.layer_list.itemChanged.connect(self.schedule_update)
        self.layer_list.model().rowsInserted.connect(self.update_layer_list_height)
        self.layer_list.model().rowsRemoved.connect(self.update_layer_list_height)
        self.layer_list.model().modelReset.connect(self.update_layer_list_height)
        ctrl_panel.addWidget(self.layer_list)

        from ui.individual import Collapsible
        self.display_settings = Collapsible("AAL 脑区显示",overlay=True)
        settings_layout = QVBoxLayout(self.display_settings.body)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.setSpacing(6)
        settings_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.display_settings.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self.display_settings.body.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self.mask_panel = QFrame()
        self.mask_panel.setVisible(False)
        mask_v = QVBoxLayout(self.mask_panel)
        mask_v.setContentsMargins(0, 0, 0, 0)
        mask_v.setSpacing(4)
        self.target_display_panel = QFrame()
        target_controls = QVBoxLayout(self.target_display_panel); target_controls.setContentsMargins(0,0,0,0); target_controls.setSpacing(4)
        self.chk_show_roi = QCheckBox("显示靶区轮廓"); self.chk_show_roi.setChecked(True)
        self.chk_apply_mask = QCheckBox("仅在靶区内显示电场")
        self.chk_apply_mask.setToolTip("按当前靶区裁剪电场显示，不改变场数据、方案或优化结果。")
        target_controls.addWidget(self.chk_show_roi); target_controls.addWidget(self.chk_apply_mask)
        self.chk_show_roi.toggled.connect(self.on_target_display_changed)
        self.chk_apply_mask.toggled.connect(self.on_target_display_changed)
        self.target_display_panel.setVisible(False); ctrl_panel.addWidget(self.target_display_panel)
        self.label_list = QListWidget()
        self.label_list.setUniformItemSizes(True)
        self.label_list.setFixedHeight(6 * (self.label_list.fontMetrics().height() + 8) + 2 * self.label_list.frameWidth() + 4)
        self.label_list.itemChanged.connect(self.on_label_selection_changed)
        label_actions = QHBoxLayout()
        label_actions.addWidget(QLabel("AAL 脑区"), 1)
        self.select_all_labels_button = QPushButton("全选")
        self.clear_all_labels_button = QPushButton("全不选")
        for button, checked in ((self.select_all_labels_button, True), (self.clear_all_labels_button, False)):
            button.setFixedHeight(28)
            button.clicked.connect(lambda _, value=checked: self.set_all_atlas_labels(value))
            label_actions.addWidget(button)
        mask_v.addLayout(label_actions)
        mask_v.addWidget(self.label_list)
        settings_layout.addWidget(self.mask_panel)

        render_ctrl = QHBoxLayout()
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(["hot", "cool", "jet", "viridis", "gray", "plasma"])
        self.cmap_combo.setCurrentText("jet")
        self.cmap_combo.currentIndexChanged.connect(self.on_cmap_changed)
        self.alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(100)
        self.alpha_slider.valueChanged.connect(self.on_alpha_changed)
        render_ctrl.addWidget(QLabel("色表:"), 0)
        render_ctrl.addWidget(self.cmap_combo, 1)
        render_ctrl.addWidget(QLabel("透明度:"), 0)
        render_ctrl.addWidget(self.alpha_slider, 1)


        self.field_range_panel = QFrame()
        range_layout = QVBoxLayout(self.field_range_panel)
        range_layout.setContentsMargins(0,0,0,0); range_layout.setSpacing(4)
        self.field_peak_label = QLabel("电场峰值：尚未计算"); range_layout.addWidget(self.field_peak_label)
        self.min_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_slider.setRange(0, 1000)
        self.min_slider.setValue(50)
        self.max_slider = QSlider(Qt.Orientation.Horizontal)
        self.max_slider.setRange(0,1000); self.max_slider.setValue(950)
        self.range_min_label = QLabel("下限 5%"); self.range_max_label = QLabel("上限 95%")
        for label,slider in ((self.range_min_label,self.min_slider),(self.range_max_label,self.max_slider)):
            row=QHBoxLayout(); row.addWidget(label); row.addWidget(slider,1); range_layout.addLayout(row)
        self.min_slider.valueChanged.connect(lambda *_args:self.on_field_range_changed('min'))
        self.max_slider.valueChanged.connect(lambda *_args:self.on_field_range_changed('max'))
        reset=QPushButton("恢复默认范围"); reset.setFixedHeight(26); reset.clicked.connect(self.reset_field_range); range_layout.addWidget(reset)
        self.field_range_panel.setVisible(False); ctrl_panel.addWidget(self.field_range_panel)

        self.display_settings.setVisible(False);ctrl_panel.addWidget(self.display_settings)

        self.info_panel = QFrame()
        self.info_panel.setObjectName("fieldInfo")
        self.info_panel.setStyleSheet("QFrame#fieldInfo { background-color: #1c1f26; border: 1px solid #353943; border-radius: 8px; } QLabel { border: none; }")
        info_layout=QVBoxLayout(self.info_panel)
        info_layout.setContentsMargins(8,4,8,4); info_layout.setSpacing(3)
        info_row = QHBoxLayout()
        info_layout.addLayout(info_row)
        info_row.setContentsMargins(8, 5, 8, 5); info_row.setSpacing(12)
        self.coord_label = QLabel("影像坐标：—")
        self.val_label = QLabel("强度: 0.0000")
        self.coord_label.setStyleSheet("color: #a8afbd; font-size: 11px;")
        self.val_label.setStyleSheet("color: #a8afbd; font-size: 11px;")
        for label in (self.coord_label,self.val_label):
            label.setMinimumWidth(0); label.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Preferred)
            info_row.addWidget(label,1)
        self.cmap_combo.setFixedWidth(120)
        self.alpha_slider.setMaximumWidth(230)
        self.alpha_value_label=QLabel('100%'); self.alpha_value_label.setFixedWidth(34)
        self.alpha_slider.valueChanged.connect(lambda value:self.alpha_value_label.setText(f'{value}%'))
        render_ctrl.addWidget(self.alpha_value_label); render_ctrl.addStretch(1)
        locator_panel = QFrame(); locator_layout = QHBoxLayout(locator_panel); locator_layout.setContentsMargins(6,0,0,0)
        self.chk_show_locator = QCheckBox("定位标记"); self.chk_show_locator.setChecked(True)
        self.chk_show_locator.setToolTip("显示或隐藏切片定位点和十字虚线；仍可点击、拖动和滚轮切片。")
        self.chk_show_locator.toggled.connect(self.on_locator_changed)
        locator_layout.addWidget(self.chk_show_locator); render_ctrl.addWidget(locator_panel)
        info_layout.addLayout(render_ctrl)
        view_layout.addWidget(self.info_panel,0,0,1,2)

        from ui.individual import Collapsible
        self.worker_panel=QFrame(); worker_layout=QVBoxLayout(self.worker_panel)
        worker_layout.setContentsMargins(0,4,0,0); worker_layout.setSpacing(3)
        worker_row=QHBoxLayout(); self.worker_status=QLabel('先选择一张 MRI，即可开始。')
        self.worker_status.setWordWrap(True); worker_row.addWidget(self.worker_status,1)
        self.cancel_worker_button=QPushButton('取消计算'); self.cancel_worker_button.setVisible(False); worker_row.addWidget(self.cancel_worker_button)
        worker_layout.addLayout(worker_row)
        self.worker_progress=QProgressBar(); self.worker_progress.setTextVisible(False); self.worker_progress.setVisible(False)
        worker_layout.addWidget(self.worker_progress)
        from ui.runtime_status import StageTimes
        self.worker_timings=StageTimes()
        self.run_details=Collapsible('运行详情'); details_layout=QVBoxLayout(self.run_details.body)
        details_layout.setContentsMargins(0,0,0,0);details_layout.addWidget(self.worker_timings)
        self.run_log=QPlainTextEdit(); self.run_log.setReadOnly(True); self.run_log.setMaximumBlockCount(1500)
        self.run_log.setFixedHeight(88); details_layout.addWidget(self.run_log); worker_layout.addWidget(self.run_details)
        self.worker_panel.setVisible(False); view_layout.addWidget(self.worker_panel,3,0,1,2)

        self.sidebar_filler_index=ctrl_panel.count(); ctrl_panel.addStretch(1)

        self.sidebar_scroll = SidebarScrollArea()
        self.sidebar_scroll.setObjectName("sidebarScroll")
        self.sidebar_scroll.setWidgetResizable(False)
        self.sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.sidebar_scroll.setMinimumWidth(400)
        self.sidebar_scroll.setMaximumWidth(680)
        self.sidebar_scroll.setWidget(ctrl_container)
        self.workspace_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.workspace_splitter.setChildrenCollapsible(False)
        self.workspace_splitter.addWidget(view_container)
        self.workspace_splitter.addWidget(self.sidebar_scroll)
        self.workspace_splitter.setStretchFactor(0, 1)
        self.workspace_splitter.setStretchFactor(1, 0)
        self.workspace_splitter.setSizes([920, 440])
        workspace_layout.addWidget(self.workspace_splitter)
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

    def on_add_patient_clicked(self, *_signal_args):
        dlg = QDialog(self)
        dlg.setWindowTitle("新建患者记录")
        dlg.setMinimumWidth(480)
        dlg.setStyleSheet("QDialog { background-color: #1c1f26; } QLabel { color: #e1e5ed; font-size: 12px; }")
        layout = QVBoxLayout(dlg)
        layout.setSpacing(10)

        title_label = QLabel("<b style='color:#c4ccd8; font-size:14px;'>患者信息录入</b>")
        layout.addWidget(title_label)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        line_style = "background-color: #181B22; color: white; border:1px solid #353943; border-radius:4px; padding:4px;"

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
        btn_box.button(QDialogButtonBox.StandardButton.Ok).setStyleSheet("background-color: #326bd1; color: white; padding: 6px 16px; border-radius: 6px; font-weight: bold;")
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

    def on_delete_patient_clicked(self, *_signal_args):
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
        title = QLabel("📋 患者记录管理")
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

        btn_back = QPushButton("← 返回首页")
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
                border: 1px solid #353943;
                border-radius: 8px;
                color: #e1e5ed;
                font-size: 13px;
            }
            QTreeWidget::item {
                padding: 6px 4px;
                border-bottom: 1px solid #353943;
            }
            QTreeWidget::item:selected {
                background-color: #326bd1;
            }
            QHeaderView::section {
                background-color: #282c35;
                color: #c4ccd8;
                padding: 8px 4px;
                border: none;
                border-bottom: 2px solid #353943;
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

    # ==================== 个体精准模式按钮回调 ====================
    def on_individual_mode_clicked(self, *_signal_args):
        self.activate_individual_mode()
        self.open_individual_workstation()

    def activate_individual_mode(self):
        self.workspace_caption.setText("刺激方案规划")
        self.individual_host.setVisible(True)
        self.ctrl_layout.setStretch(self.sidebar_filler_index, 0)
        self.worker_panel.setVisible(True)


    # ==================== 仿真报告导出 ====================

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
        self._report_eds["disease"].setText(p.get("disease_type", ""))

    def on_export_report_clicked(self, *_signal_args):
        if self.engine.data is None:
            QMessageBox.warning(self, "缺少底图", "请先加载解剖底图再导出报告。")
            return

        if not self._individual_result:
            QMessageBox.warning(self, "尚未计算电场", "请先完成方案计算，再导出仿真报告。")
            return
        disease = ""
        target = "tES 多电极" if self._individual_result.get('stimulation_type') == 'tES' else "TI 双电极对"
        if self._individual_result.get("roi"):
            target += " / ROI"

        # ── 弹窗收集信息 ──
        dlg = QDialog(self)
        dlg.setWindowTitle("仿真报告信息填写")
        dlg.setMinimumWidth(520)
        dlg.setStyleSheet("QDialog { background-color: #1c1f26; } QLabel { color: #e1e5ed; font-size: 12px; }")
        layout = QVBoxLayout(dlg)
        layout.setSpacing(10)

        line_style = "background-color: #181B22; color: white; border:1px solid #353943; border-radius:4px; padding:4px;"

        # ── 关联已有患者记录 ──
        patients = self._load_patients()
        patient_label = QLabel("<b style='color:#c4ccd8; font-size:13px;'>患者基本信息</b>")
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

        # 疾病类型由患者记录填写
        ed_disease_type = QLineEdit()
        ed_disease_type.setText(disease)
        ed_disease_type.setStyleSheet(line_style + " color: #a8afbd;")
        form_patient.addRow("患病类型:", ed_disease_type)

        layout.addLayout(form_patient)

        # 存储引用供 combo 回调使用
        self._report_eds = {
            "name": ed_name, "age": ed_age, "gender": ed_gender,
            "disease": ed_disease_type,
        }

        layout.addWidget(self._make_divider())

        # 仿真设置
        sim_label = QLabel("<b style='color:#c4ccd8; font-size:13px;'>仿真设置</b>")
        layout.addWidget(sim_label)
        form_sim = QFormLayout()
        form_sim.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        ed_target = QLineEdit()
        ed_target.setText(target)
        ed_target.setReadOnly(True)
        ed_target.setStyleSheet(line_style + " color: #a8afbd;")
        form_sim.addRow("靶区:", ed_target)

        ed_current = QLineEdit()
        ed_current.setPlaceholderText("例如 2 mA")
        if self.current_stimulation:
            ed_current.setText(self.current_stimulation)
            ed_current.setReadOnly(True)
        ed_current.setStyleSheet(line_style)
        form_sim.addRow("刺激电流:", ed_current)

        layout.addLayout(form_sim)

        from ui.result_text import FREQUENCY_NOTE
        note=QLabel(FREQUENCY_NOTE);note.setWordWrap(True);layout.addWidget(note)

        # 确认 / 取消
        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btn_box.button(QDialogButtonBox.StandardButton.Ok).setText("确认生成报告")
        btn_box.button(QDialogButtonBox.StandardButton.Ok).setStyleSheet("background-color: #326bd1; color: white; padding: 6px 16px; border-radius: 6px; font-weight: bold;")
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
            "disease_type": ed_disease_type.text().strip(),
            "current": ed_current.text().strip(),
            "slice_indices": (np.linalg.inv(self.engine.affine) @ np.r_[self.current_cursor_ras(),1])[:3].tolist(),
            "stimulation": self._individual_result,
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
        line.setStyleSheet("background-color: #353943; max-height: 1px;")
        return line

    # ==================== 电极图 ====================

    # ==================== 底图加载 ====================

    # ==================== 其他辅助方法 ====================
    def update_layer_list_height(self, *args):
        row_height = max(28, self.layer_list.sizeHintForRow(0))
        visible=sum(not self.layer_list.item(i).isHidden() for i in range(self.layer_list.count()))
        self.layer_list.setFixedHeight(max(42, min(4, visible) * row_height + 4))

    def refresh_layer_list_ui(self, schedule=True):
        self.layer_list.blockSignals(True)
        self.layer_list.clear()
        for ov in self.engine.overlays:
            name = ov.get("display_name", ("脑区：" if ov.get("is_atlas") else "") + ov["name"])
            item = QListWidgetItem(name)
            item.setCheckState(Qt.CheckState.Checked)
            self.layer_list.addItem(item)
            item.setHidden(bool(ov.get('is_roi')))
        visible=[i for i in range(self.layer_list.count()) if not self.layer_list.item(i).isHidden()]
        if visible:self.layer_list.setCurrentRow(visible[-1])
        self.update_layer_list_height()
        self.layer_list.blockSignals(False)
        self.sync_cmap_selector(self.layer_list.currentRow())
        if schedule:
            self.schedule_update()




    def execute_base_mri_load(self, filepath):
        filepath = os.fspath(filepath)
        replacement = NiiEngine()
        success, info = replacement.load_image(filepath)
        if not success:
            QMessageBox.warning(self, "导入失败", f"错误解析: {info}")
            return False
        self.timer_2d.stop(); self.timer_3d.stop()
        self._drag_active = False
        for canvas in self.canvases.values():
            canvas._dragging = False
        self.engine = replacement
        self.engine.show_locator = self.chk_show_locator.isChecked()
        self.engine.show_roi = self.chk_show_roi.isChecked()
        self.engine.clip_fields_to_roi = False
        self.chk_apply_mask.blockSignals(True); self.chk_apply_mask.setChecked(False); self.chk_apply_mask.blockSignals(False)
        self.target_display_panel.hide()
        self.current_stimulation = None
        self._individual_result = None
        self.current_label.setText("刺激电流：尚未合成")
        from ui.result_text import FREQUENCY_NOTE
        self.frequency_note.setText(FREQUENCY_NOTE)
        dialog = getattr(self, "individual_dialog", None)
        if dialog and (not dialog.task or os.path.abspath(filepath) != os.path.abspath(dialog.task.get("padded_t1", ""))):
            dialog.t1.setText(filepath)
        self.layer_list.clear()
        self.label_list.clear()
        self.mask_panel.setVisible(False)
        self.field_range_panel.setVisible(False)
        self.display_settings.set_expanded(False); self.display_settings.setVisible(False)
        self.canvas_3d.clear_volume()
        self.current_base_path = filepath
        self.workspace_file.setText(os.path.basename(filepath).replace('T1_ras_padded.nii','模型 T1 MRI'))
        self._last_browse_dir = os.path.dirname(os.path.abspath(filepath))
        sh = self.engine.view.shape
        for i, a in enumerate(["X", "Y", "Z"]):
            self.sliders[a].setRange(0, sh[i] - 1)
            self.sliders[a].setValue(sh[i] // 2)
        self.schedule_update()
        return True


    def execute_overlay_mri_load(self, filepath):
        if self.engine.data is None:
            QMessageBox.warning(self, "无法叠加", "无基础底图，请先载入背景。")
            return
        success, info = self.engine.load_overlay(filepath)
        if not success:
            QMessageBox.warning(self, "叠加失败", f"错误: {info}")
            return
        self.refresh_layer_list_ui()








    def open_individual_workstation(self):
        if not hasattr(self, "individual_dialog"):
            from ui.individual import IndividualPanel
            self.individual_dialog = IndividualPanel(self, self.individual_host)
            self.individual_host_layout.addWidget(self.individual_dialog)
            self.individual_dialog.task_ready.connect(self.on_individual_task_ready)
            self.individual_dialog.field_ready.connect(self.on_individual_field_ready)
            self.individual_dialog.atlas_ready.connect(self.on_individual_atlas_ready)
        self.stack.setCurrentIndex(1)
        self.individual_host.show()
        self.individual_dialog.show()

    def on_individual_atlas_ready(self, record):
        self.statusBar().showMessage("AAL3 已映射到个体 MRI，可搜索并选择脑区。",5000)

    def current_cursor_ras(self):
        return self.engine.view.voxel_to_native_ras([self.sliders[a].value() for a in ('X','Y','Z')])

    def locate_native_voxel(self, center):
        view_center = self.engine.view.native_voxel_to_view(center)
        for index, axis in enumerate(('X','Y','Z')):
            slider = self.sliders[axis]
            slider.blockSignals(True)
            slider.setValue(int(np.clip(round(view_center[index]), 0, self.engine.view.shape[index]-1)))
            slider.blockSignals(False)
        self.schedule_update()

    def on_locator_changed(self, checked):
        self.engine.show_locator = bool(checked)
        self.update_2d_views()

    def on_target_display_changed(self, *args):
        self.engine.clip_fields_to_roi = self.chk_apply_mask.isChecked()
        self.engine.show_roi = self.chk_show_roi.isChecked()
        for index, overlay in enumerate(self.engine.overlays):
            if overlay.get('is_roi') and index < self.layer_list.count():
                self.layer_list.item(index).setCheckState(Qt.CheckState.Checked if self.chk_show_roi.isChecked() else Qt.CheckState.Unchecked)
        self.update_2d_views()
        self.update_3d_view()

    def set_target_roi(self, roi, name='靶区轮廓'):
        if self.engine.data is None: return
        shape=self.engine.data.shape
        mask=np.zeros(shape,dtype=bool)
        if roi and roi.get('mask'):
            import nibabel as nib
            image=nib.load(roi['mask'])
            if image.shape != shape or not np.allclose(image.affine,self.engine.affine,atol=1e-4): return
            values=np.asarray(image.dataobj)
            mask=np.isin(values,roi['labels']) if 'labels' in roi else values>0
        elif roi and roi.get('center_ras_mm') is not None:
            center=np.asarray(roi['center_ras_mm'],dtype=float)
            radius=float(roi['radius_mm'])
            voxel=np.linalg.solve(self.engine.affine,np.r_[center,1])[:3]
            reach=radius/max(1e-6,np.linalg.svd(self.engine.affine[:3,:3],compute_uv=False).min())
            lo=np.maximum(0,np.floor(voxel-reach).astype(int)); hi=np.minimum(shape,np.ceil(voxel+reach).astype(int)+1)
            if np.all(hi>lo):
                points=np.indices(tuple(hi-lo)).reshape(3,-1).T+lo
                world=points@self.engine.affine[:3,:3].T+self.engine.affine[:3,3]
                local=(np.sum((world-center)**2,axis=1)<=radius**2).reshape(tuple(hi-lo))
                mask[tuple(slice(int(a),int(b)) for a,b in zip(lo,hi))]=local
        flags=[self.layer_list.item(i).checkState() for i in range(self.layer_list.count())]
        index=next((i for i,ov in enumerate(self.engine.overlays) if ov.get('is_roi')),None)
        values={'name':name,'display_name':name,'data':mask.astype(np.float32),'is_roi':True,
                'is_atlas':False,'labels':[],'active_labels':list((roi or {}).get('labels',[])),
                'label_dict':{},'min':0.,'max':1.,'data_min':0.,'data_max':1.,
                'display_min':0.,'display_max':1.,'cmap':'jet','alpha':1.,'units':''}
        if index is None:
            if not mask.any(): return
            index=len(self.engine.overlays);self.engine.overlays.append(values)
        else:self.engine.overlays[index].update(values)
        self.refresh_layer_list_ui(schedule=False)
        for i,state in enumerate(flags):self.layer_list.item(i).setCheckState(state)
        self.layer_list.item(index).setCheckState(Qt.CheckState.Checked if mask.any() and self.chk_show_roi.isChecked() else Qt.CheckState.Unchecked)
        self.target_display_panel.setVisible(bool(mask.any()))
        if not mask.any():
            self.chk_apply_mask.blockSignals(True);self.chk_apply_mask.setChecked(False);self.chk_apply_mask.blockSignals(False)
            self.engine.clip_fields_to_roi=False
        self.update_2d_views();self.update_3d_view()

    def preview_atlas_regions(self,record,labels):
        self.set_target_roi({'mask':record['atlas_path'],'labels':list(labels)},'靶区轮廓 · AAL3')
        if not labels:return
        target=next((ov for ov in reversed(self.engine.overlays) if ov.get('is_roi')),None)
        if target is None:return
        selected=np.argwhere(target['data']>0)
        if len(selected):
            center=selected.mean(axis=0)
            self.locate_native_voxel(center)
        self.schedule_update()

    def on_individual_task_ready(self, task):
        self.activate_individual_mode()
        self.execute_base_mri_load(task["padded_t1"])
        self.individual_task = task
        self.execute_overlay_mri_load(task["tissue_masks"])
        if self.engine.overlays:
            self.engine.overlays[-1]["name"] = "头部组织（可选）"
            self.engine.overlays[-1]["display_name"] = "头部组织（可选）"
            self.refresh_layer_list_ui(schedule=False)
        if self.layer_list.count():
            self.layer_list.item(self.layer_list.count() - 1).setCheckState(Qt.CheckState.Unchecked)
        self.update_2d_views()
        self.current_label.setText("刺激电流：完成方案计算后显示")

    def on_individual_field_ready(self, result):
        try:
            clip_to_target=self.chk_apply_mask.isChecked()
            padded = result.get("padded_t1")
            if not padded or not os.path.isfile(padded):
                raise ValueError("结果缺少有效的个体 padded T1")
            self.activate_individual_mode()
            self.execute_base_mri_load(padded)
            names = ["tES 电场幅值（峰值）(V/m)" if result.get('stimulation_type')=='tES' else "TI 最大调制包络（峰值）(V/m)"]
            for i, overlay in enumerate(result["overlays"][:1]):
                ok, info = self.engine.load_overlay(overlay,kind='field')
                if not ok:
                    raise ValueError(info)
                self.engine.overlays[-1]["name"] = names[i]
                self.engine.overlays[-1]["cmap"] = "jet"
            if result.get("roi"):
                roi_path = os.path.join(os.path.dirname(result["overlays"][0]), "ROI.nii.gz")
                if os.path.isfile(roi_path):
                    ok, _ = self.engine.load_overlay(roi_path)
                    if ok:
                        self.engine.overlays[-1]["display_name"] = "靶区轮廓"
                        self.engine.overlays[-1]["is_roi"] = True
                        self.engine.overlays[-1]["is_atlas"] = False
                        self.engine.overlays[-1]["labels"] = []; self.engine.overlays[-1]["active_labels"] = []
                        self.target_display_panel.show()
                        if result['roi'].get('center_ras_mm'):
                            center=(np.linalg.inv(self.engine.affine) @ np.r_[result['roi']['center_ras_mm'],1])[:3]
                        else:
                            points=np.argwhere(self.engine.overlays[-1]['data']>0); center=points.mean(axis=0) if len(points) else None
                        if center is not None:
                            self.locate_native_voxel(center)
            self.refresh_layer_list_ui(schedule=False)
            for i in range(1, self.layer_list.count()):
                self.layer_list.item(i).setCheckState(Qt.CheckState.Checked if self.engine.overlays[i].get('is_roi') and self.chk_show_roi.isChecked() else Qt.CheckState.Unchecked)
            self.layer_list.setCurrentRow(0)
            self.chk_apply_mask.setChecked(clip_to_target)
            self.engine.clip_fields_to_roi = self.chk_apply_mask.isChecked()
            from ui.individual import current_description, electrode_names, model_root
            self.current_stimulation = current_description(result, electrode_names(model_root()))
            self._individual_result = result
            from ui.result_text import stimulation_note
            note=stimulation_note(result)
            self.frequency_note.setText('tES：直流值或同频交流峰值（0°/180°）。频率设定需要结合实际情况。' if result.get('stimulation_type')=='tES' else note)
            self.frequency_note.setToolTip(note)
            task_dir = os.path.dirname(result["prediction_dir"])
            manifest_path = os.path.join(task_dir, "manifest.json")
            if os.path.isfile(manifest_path):
                with open(manifest_path, encoding="utf-8") as stream:
                    manifest = json.load(stream)
                self.individual_task = {"electrodes": manifest["stages"]["electrodes"], "task_dir": task_dir}
            from ui.result_text import electrode_scheme_html,roi_statistics_text
            self.current_label.setText(electrode_scheme_html(result,electrode_names(model_root()))+
                '<br><span style="color:#ffd38b;font-size:11px">'+roi_statistics_text(result)+'</span>')
            self.update_2d_views()
            self.schedule_update()
        except Exception as exc:
            QMessageBox.warning(self, "个体电场加载失败", str(exc))

    def show_individual_electrodes(self, *_signal_args):
        task = getattr(self, "individual_task", None)
        path = task.get("electrodes", {}).get("electrode_coordinates") if task else None
        if not path or not os.path.isfile(path):
            QMessageBox.information(self, "个体电极", "请先完成或打开个体导联场任务。")
            return
        from ui.electrode_map import electrode_dialog
        dialog=electrode_dialog(self,path,self._individual_result)
        dialog.exec()

    def locate_roi(self,roi,padded_t1=None):
        if padded_t1 and os.path.abspath(self.current_base_path or '')!=os.path.abspath(padded_t1):
            self.execute_base_mri_load(padded_t1)
        if self.engine.data is None:raise ValueError('请先加载 MRI。')
        if roi.get('center_ras_mm'):
            center=(np.linalg.inv(self.engine.affine) @ np.r_[roi['center_ras_mm'],1])[:3]
        else:
            import nibabel as nib
            image=nib.load(roi['mask'])
            if image.shape!=self.engine.data.shape or not np.allclose(image.affine,self.engine.affine,atol=1e-4):raise ValueError('靶区文件与当前个体模型网格不一致。')
            data=np.asarray(image.dataobj)
            mask=np.isin(data,roi['labels']) if roi.get('labels') else data>0
            points=np.argwhere(mask)
            if not len(points):raise ValueError('选定靶区为空。')
            center=points.mean(axis=0)
        self.locate_native_voxel(center)
        self.update_2d_views();self.update_3d_view()

    def closeEvent(self, event):
        self.timer_2d.stop(); self.timer_3d.stop()
        dialog = getattr(self, "individual_dialog", None)
        if dialog and dialog.process:
            dialog.cancel()
            dialog.process.waitForFinished(3000)
        self.canvas_3d.shutdown()
        super().closeEvent(event)

    def update_slice_coordinates(self):
        if not hasattr(self,'slice_coord_labels'): return
        if self.engine.data is None:
            for axis,label in self.slice_coord_labels.items():
                label.setText(f'{axis}：—'); self.sliders[axis].setEnabled(False)
            return
        values=[self.sliders[a].value() for a in ('X','Y','Z')]
        xyz=self.current_cursor_ras()
        for index,axis in enumerate(('X','Y','Z')):
            self.sliders[axis].setEnabled(True)
            text=f'{axis}：{xyz[index]:.1f} mm   ·   {values[index]+1}/{self.engine.view.shape[index]} 层'
            self.slice_coord_labels[axis].setText(text)
            self.slice_coord_labels[axis].setToolTip('RAS 坐标；'+text)

    def schedule_update(self, *_signal_args):
        self.update_slice_coordinates()
        if self.engine.data is None:
            return
        self.timer_2d.start(15)
        if not self._drag_active:
            self.timer_3d.start(500)

    def sync_cmap_selector(self, row):
        if row < 0 or row >= len(self.engine.overlays):
            self.field_range_panel.setVisible(False);self.display_settings.setVisible(False);self.display_settings.set_expanded(False)
            return
        ov = self.engine.overlays[row]
        self.display_settings.setVisible(bool(ov.get("is_atlas")) and not ov.get("is_roi"))
        if not ov.get("is_atlas") or ov.get("is_roi"):self.display_settings.set_expanded(False)
        self.field_range_panel.setVisible(not ov.get('is_atlas') and not ov.get('is_roi'))
        for slider,key in ((self.min_slider,'display_min'),(self.max_slider,'display_max')):
            slider.blockSignals(True); span=ov['max']-ov['min']
            value=ov.get(key,ov['min'] if key=='display_min' else ov['max'])
            slider.setValue(round(1000*(value-ov['min'])/span) if span else 0); slider.blockSignals(False)
        self.update_range_labels(ov)
        idx = self.cmap_combo.findText(ov["cmap"])
        if idx >= 0:
            self.cmap_combo.blockSignals(True)
            self.cmap_combo.setCurrentIndex(idx)
            self.cmap_combo.blockSignals(False)
        self.alpha_slider.blockSignals(True)
        self.alpha_slider.setValue(int(ov.get("alpha", 1.0) * 100))
        self.alpha_slider.blockSignals(False)
        self.alpha_value_label.setText(f'{self.alpha_slider.value()}%')
        if ov.get("is_atlas") and not ov.get("is_roi"):
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
            row_height = self.label_list.sizeHintForRow(0)
            if row_height > 0:
                self.label_list.setFixedHeight(6 * row_height + 2 * self.label_list.frameWidth() + 4)
        else:
            self.mask_panel.setVisible(False)

    def update_range_labels(self,ov):
        unit=ov.get('units','V/m')
        self.field_peak_label.setText(f"实际峰值 {ov.get('data_max',ov['max']):.4g} {unit}")
        self.range_min_label.setText(f"下限 {ov.get('display_min',ov['min']):.3g}")
        self.range_max_label.setText(f"上限 {ov.get('display_max',ov['max']):.3g}")

    def on_field_range_changed(self,moved):
        row=self.layer_list.currentRow()
        if not 0<=row<len(self.engine.overlays): return
        ov=self.engine.overlays[row]
        if ov.get('is_atlas'): return
        low,high=self.min_slider.value(),self.max_slider.value()
        if low>=high:
            slider=self.min_slider if moved=='min' else self.max_slider
            slider.blockSignals(True); slider.setValue(max(0,high-1) if moved=='min' else min(1000,low+1)); slider.blockSignals(False)
            low,high=self.min_slider.value(),self.max_slider.value()
        span=ov['max']-ov['min']
        ov['display_min']=ov['min']+span*low/1000; ov['display_max']=ov['min']+span*high/1000
        self.update_range_labels(ov); self.schedule_update()

    def reset_field_range(self, *_signal_args):
        row=self.layer_list.currentRow()
        if not 0<=row<len(self.engine.overlays): return
        ov=self.engine.overlays[row]
        ov['display_min']=.05*ov['max']; ov['display_max']=.95*ov['max']
        self.sync_cmap_selector(row); self.schedule_update()

    def on_cmap_changed(self, *_signal_args):
        row = self.layer_list.currentRow()
        if row >= 0 and row < len(self.engine.overlays):
            self.engine.overlays[row]["cmap"] = self.cmap_combo.currentText()
            self.schedule_update()

    def on_alpha_changed(self, *_signal_args):
        row = self.layer_list.currentRow()
        if row >= 0 and row < len(self.engine.overlays):
            self.engine.overlays[row]["alpha"] = self.alpha_slider.value() / 100.0
            self.schedule_update()

    def set_all_atlas_labels(self, checked):
        row = self.layer_list.currentRow()
        if not 0 <= row < len(self.engine.overlays) or not self.engine.overlays[row].get('is_atlas'): return
        self.label_list.blockSignals(True)
        for index in range(self.label_list.count()):
            self.label_list.item(index).setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        self.label_list.blockSignals(False)
        self.on_label_selection_changed()

    def on_label_selection_changed(self, *_signal_args):
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
        sh = self.engine.view.shape
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
        """3D视图底部滑块拖动 → 同步到侧边栏滑块并刷新2D"""
        if self.engine.data is None:
            return
        sh = self.engine.view.shape
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
        dialog = getattr(self, "individual_dialog", None)
        if dialog and dialog.selection_mode:
            for canvas in self.canvases.values():
                canvas._dragging = False
            self._drag_active = False
            dialog.capture_target_from_viewer()

    def handle_canvas_scroll(self, plane, steps):
        if self.engine.data is None:
            return
        axis = {'sagittal': 'X', 'coronal': 'Y', 'axial': 'Z'}.get(plane)
        if axis:
            slider = self.sliders[axis]
            slider.setValue(max(slider.minimum(), min(slider.maximum(), slider.value() + steps)))

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
            target=next((ov for ov in reversed(self.engine.overlays) if ov.get('is_roi')),None)
            if target is not None:
                atlas_vol,active_labels=self.engine.view.overlay_data(target)>0,[1]
        active_ovs = []
        for i in range(self.layer_list.count()):
            if i < len(self.engine.overlays) and self.layer_list.item(i).checkState() == Qt.CheckState.Checked:
                ov = self.engine.overlays[i]
                if ov.get("is_roi") and not self.chk_show_roi.isChecked():continue
                d_range = ov["max"] - ov["min"]
                cur_min = ov.get('display_min',ov['min'])
                active_ovs.append({
                    "data": self.engine.view.overlay_data(ov), "is_atlas": ov.get("is_atlas", False),
                    "active_labels": ov.get("active_labels", []), "cmap": ov["cmap"],
                    "alpha": ov.get("alpha", 1.0), "min": cur_min, "max": ov.get('display_max',ov['max']),
                    "name": ov["name"], "units":ov.get('units',''), "is_roi":ov.get('is_roi',False)
                })
        return x, y, z, atlas_vol, active_labels, active_ovs

    def update_2d_views(self):
        self.update_slice_coordinates()
        if self.engine.data is None:
            return
        x, y, z, atlas_vol, active_labels, active_ovs = self.get_render_params()
        mni = self.current_cursor_ras()
        self.coord_label.setText(f"RAS: ({mni[0]:.1f}, {mni[1]:.1f}, {mni[2]:.1f}) mm")
        ov_info = ""
        for ov in active_ovs:
            if ov['is_atlas'] or ov.get('is_roi'): continue
            sh = ov["data"].shape
            val = float(ov["data"][x, y, z]) if x < sh[0] and y < sh[1] and z < sh[2] else 0.0
            ov_info += f" | {ov['name'][:8]}: {val:.3f}"
        self.val_label.setText(f"底图: {float(self.engine.view.data[x, y, z]):.1f}{ov_info}")
        self.coord_label.setToolTip(self.coord_label.text()); self.val_label.setToolTip(self.val_label.text())

        sx, sy, sz = np.linalg.norm(self.engine.view.affine[:3,:3],axis=0)
        sag_ovs, cor_ovs, axi_ovs = [], [], []
        for ov in active_ovs:
            sag_slc, cor_slc, axi_slc = ov["data"][x, :, :], ov["data"][:, y, :], ov["data"][:, :, z]
            if ov["is_atlas"]:
                sag_slc = np.where(np.isin(sag_slc, ov["active_labels"]), sag_slc, 0)
                cor_slc = np.where(np.isin(cor_slc, ov["active_labels"]), cor_slc, 0)
                axi_slc = np.where(np.isin(axi_slc, ov["active_labels"]), axi_slc, 0)
            elif atlas_vol is not None and not ov.get('is_roi'):
                sag_slc = np.where(np.isin(atlas_vol[x, :, :], active_labels), sag_slc, np.nan)
                cor_slc = np.where(np.isin(atlas_vol[:, y, :], active_labels), cor_slc, np.nan)
                axi_slc = np.where(np.isin(atlas_vol[:, :, z], active_labels), axi_slc, np.nan)
            sag_ovs.append({**ov, "data": sag_slc})
            cor_ovs.append({**ov, "data": cor_slc})
            axi_ovs.append({**ov, "data": axi_slc})

        self.canvases["sagittal"].render_slice(self.engine.view.data[x, :, :], aspect=sz / sy, crosshair=(y, z) if self.chk_show_locator.isChecked() else None, overlays=sag_ovs, vmin=self.engine.display_min, vmax=self.engine.display_max)
        self.canvases["coronal"].render_slice(self.engine.view.data[:, y, :], aspect=sz / sx, crosshair=(x, z) if self.chk_show_locator.isChecked() else None, overlays=cor_ovs, vmin=self.engine.display_min, vmax=self.engine.display_max)
        self.canvases["axial"].render_slice(self.engine.view.data[:, :, z], aspect=sy / sx, crosshair=(x, y) if self.chk_show_locator.isChecked() else None, overlays=axi_ovs, vmin=self.engine.display_min, vmax=self.engine.display_max)

    def update_3d_view(self):
        if self.engine.data is None:
            return
        x,y,z,atlas_vol,active_labels,active_ovs=self.get_render_params()
        overlays=[]
        for ov in active_ovs:
            data=ov['data']
            if ov['is_atlas']:data=np.where(np.isin(data,ov['active_labels']),data,0)
            elif atlas_vol is not None and not ov.get('is_roi'):data=np.where(np.isin(atlas_vol,active_labels),data,np.nan)
            overlays.append({**ov,'volume':data})
        try:
            self.canvas_3d.render_3d(self.engine.view.data,x,y,z,overlays=overlays,
                vmin=self.engine.display_min,vmax=self.engine.display_max,affine=self.engine.view.affine)
        except Exception as exc:
            print(f"3D render skipped: {exc}")


def main():
    if sys.platform == 'win32':
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('NeuroGuide.Desktop')
    app = QApplication(sys.argv)
    app.setApplicationName('NeuroGuide')
    app.setApplicationVersion('4.5')
    ex = ZjuMriViewer()
    startup_check = "--startup-check" in sys.argv
    if startup_check:
        if '--field-check' in sys.argv:
            from tools.release_checks import check_field_result
            check_field_result(ex,sys.argv[sys.argv.index('--field-check')+1])
        else:
            ex.on_individual_mode_clicked()
    if not startup_check:
        ex.show()
    app.processEvents()
    ready_path = os.environ.get("NEUROGUIDE_STARTUP_READY")
    if ready_path:
        with open(ready_path, "w", encoding="utf-8") as stream:
            stream.write("ready")
    if startup_check:
        ex.close()
        sys.exit(0)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
