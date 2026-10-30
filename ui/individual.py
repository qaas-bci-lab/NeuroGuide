"""Guided individual planning UI; runtime configuration stays internal."""
from __future__ import annotations
import codecs
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from uuid import uuid4
from PyQt6.QtCore import QProcess, Qt, pyqtSignal, QTimer
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QComboBox, QDoubleSpinBox, QSpinBox, QTabWidget, QWidget,
    QPlainTextEdit, QMessageBox, QScrollArea, QFrame, QToolButton, QProgressBar, QListWidget, QListWidgetItem, QSizePolicy, QStackedWidget)

def app_root():
    return Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parents[1]

def model_root():
    external = app_root() / 'BrainLFNet'
    return external if external.is_dir() else Path(getattr(sys, '_MEIPASS', app_root())) / 'BrainLFNet'

def task_root():
    return app_root() / 'NeuroGuide/individualized'

def atlas_root():
    return Path(getattr(sys,'_MEIPASS',Path(__file__).resolve().parents[1]))/'assets/atlases'

def builtin_mri_path():
    """Resolve the reference MRI in source and frozen resource layouts."""
    roots = [app_root(), Path(getattr(sys, '_MEIPASS', app_root()))]
    for root in dict.fromkeys(roots):
        for name in ('TI.nii', 'T1.nii', 'TI.nii.gz', 'T1.nii.gz'):
            candidate = root / 'mri' / name
            if candidate.is_file():
                return candidate
    raise FileNotFoundError('找不到内置 T1 MRI，请检查程序的 mri 文件夹。')

def worker_command(root, config_path):
    if getattr(sys, 'frozen', False):
        return str(Path(root) / 'NeuroGuideWorker.exe'), ['--individual-worker', str(config_path)]
    interpreter = Path(sys.executable)
    if interpreter.name.lower() == 'pythonw.exe':
        interpreter = interpreter.with_name('python.exe')
    return str(interpreter), ['-u', str(Path(root) / 'tools/individual_worker.py'), str(config_path)]

def electrode_names(brain_root):
    path = Path(brain_root) / 'SegandElec/assets/elec72.loc'
    if not path.is_file():
        return {i: f'E{i}' for i in range(1, 73)}
    return {int(p[0]): p[3].rstrip('.') for line in path.read_text(encoding='utf-8').splitlines() if len(p := line.split()) >= 4}

def current_description(result, names=None):
    names = names or {}
    if result.get('stimulation_type')=='tES':
        return 'tES：'+'；'.join(f'{names.get(label,label)} {float(c):+.4f} mA（峰值）' for label,c in zip(result['electrode_labels'],result['currents_ma']))
    return '；'.join(f'通道{k+1} {names.get(p[0], p[0])}→{names.get(p[1], p[1])}: ±{float(c):g} mA 峰值'
                    for k, (p, c) in enumerate(zip(result['pairs'], result['currents_peak_ma'])))

class Collapsible(QFrame):
    def __init__(self, title, parent=None, overlay=False):
        super().__init__(parent)
        self.overlay=overlay;self.drawer=None
        layout = QVBoxLayout(self); layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(8)
        self.toggle = QToolButton(); self.toggle.setText(title); self.toggle.setCheckable(True)
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(Qt.ArrowType.RightArrow); self.toggle.setObjectName('disclosure'); layout.addWidget(self.toggle)
        self.body = QFrame(); self.body.setVisible(False); layout.addWidget(self.body)
        self.toggle.toggled.connect(self.set_expanded)

    def set_expanded(self, expanded):
        self.toggle.blockSignals(True);self.toggle.setChecked(expanded);self.toggle.blockSignals(False)
        if self.overlay:
            if expanded:
                from ui.drawer import SidebarDrawer
                if self.drawer is None:self.drawer=SidebarDrawer(self)
                self.drawer.reveal()
            elif self.drawer:self.drawer.hide()
            else:self.body.hide()
        else:self.body.setVisible(expanded)
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)

    def hideEvent(self,event):
        if self.overlay:self.set_expanded(False)
        super().hideEvent(event)

class CompactDoubleSpinBox(QDoubleSpinBox):
    def textFromValue(self, value):
        text=f'{value:.{self.decimals()}f}'
        if self.decimals()>0:text=text.rstrip('0').rstrip('.')
        return '0' if text in {'','-0'} else text

class IndividualPanel(QWidget):
    task_ready = pyqtSignal(object)
    field_ready = pyqtSignal(object)
    atlas_ready = pyqtSignal(object)

    def __init__(self, owner, parent=None):
        super().__init__(parent or owner)
        self.owner = owner
        self.timings=owner.worker_timings
        self.setObjectName('individualPanel'); self.setMinimumHeight(280)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.root, self.storage_root, self.models = app_root(), task_root(), model_root()
        self.process = None; self._result = self._error = None; self._buffer = ''; self._cancelled = False; self._log_stream = None
        self.task = None; self.selection_mode = False; self.target_selected = False
        self.atlas_record = None; self.atlas_resources = atlas_root()
        from ui.theme import PANEL_STYLE
        self.setStyleSheet(PANEL_STYLE)
        layout = QVBoxLayout(self); layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(6)
        layout.addWidget(self.label('刺激方案', 'heading'))
        mode_row=QHBoxLayout();mode_row.addWidget(self.label('刺激方式'))
        self.stimulation_mode=QComboBox();self.stimulation_mode.addItems(['TI · 双电极对','tES · 多电极'])
        mode_row.addWidget(self.stimulation_mode,1);layout.addLayout(mode_row)
        from ui.workflow_tabs import WorkflowTabs
        self.tabs = WorkflowTabs(); layout.addWidget(self.tabs, 1)
        self.build_import_page(); self.build_optimization_page(); self.build_manual_page()
        self.stimulation_mode.currentIndexChanged.connect(self.update_stimulation_mode)
        self.action_stack = QStackedWidget()
        for button in (self.predict_button, self.optimize_button, self.synthesize_button): self.action_stack.addWidget(button)
        self.action_stack.setFixedHeight(34); self.tabs.set_actions(self.action_stack)
        self.tabs.currentChanged.connect(self.stop_target_selection)
        self.tabs.currentChanged.connect(self.show_shared_target_controls)
        self.status=owner.worker_status; self.progress=owner.worker_progress
        self.details=owner.run_details; self.log=owner.run_log; self.cancel_button=owner.cancel_worker_button
        self.cancel_button.clicked.connect(self.cancel)
        self.status.setText('先选择一张 MRI，即可开始。')
        self.populate_regions(); self.roi_mode.setCurrentIndex(1)
        self.update_roi_mode(); self.update_stimulation_mode(); self.update_current_preview(); self.set_busy(False)

    @staticmethod
    def label(text, name=None):
        label = QLabel(text); label.setWordWrap(True)
        if name: label.setObjectName(name)
        return label

    @staticmethod
    def card(title=None):
        card = QFrame(); card.setObjectName('card'); layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 10, 12, 10); layout.setSpacing(8)
        if title:
            label = QLabel(title); label.setStyleSheet('font-size: 13px; font-weight: bold; color: #f1f5ff;'); layout.addWidget(label)
        return card, layout

    @staticmethod
    def number(minimum, maximum, value, suffix='', decimals=2):
        control = CompactDoubleSpinBox(); control.setRange(minimum, maximum); control.setDecimals(decimals)
        control.setValue(value); control.setSuffix(suffix); return control

    @staticmethod
    def integer(minimum, maximum, value):
        control = QSpinBox(); control.setRange(minimum, maximum); control.setValue(value); return control

    def add_page(self, page, title):
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(page); self.tabs.addTab(scroll, title)

    def build_import_page(self):
        page = QWidget(); layout = QVBoxLayout(page); layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(14)
        card, body = self.card('1 · 选择 T1 MRI')
        self.t1 = QLineEdit(self); self.t1.setReadOnly(True); self.t1.setVisible(False)
        self.t1_name=self.label('尚未选择 MRI 文件')
        self.t1.textChanged.connect(self.update_mri_selection)
        if self.owner.current_base_path:
            self.t1.setText(self.owner.current_base_path)
        row = QHBoxLayout(); row.addWidget(self.t1_name, 1)
        self.browse_button = QPushButton('选择…'); self.browse_button.clicked.connect(self.browse_t1); row.addWidget(self.browse_button)
        self.builtin_button = QPushButton('使用内置 MRI')
        self.builtin_button.setToolTip('没有自己的 MRI 时，使用内置参考影像进行建模和方案探索。')
        self.builtin_button.clicked.connect(self.use_builtin_mri)
        row.addWidget(self.builtin_button)
        body.addLayout(row)
        self.dicom_button = QPushButton('导入 DICOM…')
        self.dicom_button.clicked.connect(self.import_dicom)
        body.addWidget(self.dicom_button)
        self.source_note = self.label('支持 T1 MRI（.nii / .nii.gz），或导入 DICOM 并确认 T1 序列。没有 MRI 时可使用内置影像。', 'muted')
        body.addWidget(self.source_note)
        self.predict_button = QPushButton('生成电场模型'); self.predict_button.setObjectName('primary')
        self.predict_button.clicked.connect(self.start_prediction)
        self.task_label = self.label('选择 MRI 后会检查已有模型；首次生成需要一些时间。', 'muted')
        body.addWidget(self.task_label); layout.addWidget(card)
        layout.addStretch(); self.add_page(page, '1 影像')

    def build_optimization_page(self):
        page = QWidget(); layout = QVBoxLayout(page); layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(14)
        self.shared_target_controls=QWidget()
        self.shared_target_layout=QVBoxLayout(self.shared_target_controls);self.shared_target_layout.setContentsMargins(0,0,0,0);self.shared_target_layout.setSpacing(8)
        self.optimization_target_host=QVBoxLayout();layout.addLayout(self.optimization_target_host)
        self.optimization_target_host.addWidget(self.shared_target_controls)
        card, body = self.card('选择靶区')
        self.roi_mode = QComboBox(); self.roi_mode.addItems(['在 MRI 上点选','按 AAL3 脑区选择'])
        roi_actions=QHBoxLayout();roi_actions.addWidget(self.roi_mode,1)
        self.locate_button=QPushButton('定位靶区');self.locate_button.setToolTip('将三个切片定位到靶区中心。');self.locate_button.clicked.connect(self.locate_target)
        roi_actions.addWidget(self.locate_button);body.addLayout(roi_actions)
        self.target_label = self.label('尚未选择目标位置', 'muted'); body.addWidget(self.target_label)
        self.atlas_panel = QWidget(); atlas_layout = QVBoxLayout(self.atlas_panel); atlas_layout.setContentsMargins(0,0,0,0); atlas_layout.setSpacing(6)
        self.atlas_status = self.label('先准备个体脑区图谱，之后可搜索并勾选脑区。','muted'); atlas_layout.addWidget(self.atlas_status)
        self.atlas_prepare = QPushButton('准备 AAL3 脑区图谱'); self.atlas_prepare.clicked.connect(self.start_atlas); atlas_layout.addWidget(self.atlas_prepare)
        self.region_search = QLineEdit(); self.region_search.setPlaceholderText('搜索脑区，例如 海马 / hippocampus'); atlas_layout.addWidget(self.region_search)
        self.region_search.textChanged.connect(self.filter_regions)
        self.region_list = QListWidget(); self.region_list.setFixedHeight(130); atlas_layout.addWidget(self.region_list)
        self.region_list.itemChanged.connect(self.regions_changed)
        self.clear_regions_button = QPushButton('清空脑区选择'); self.clear_regions_button.clicked.connect(self.clear_regions); atlas_layout.addWidget(self.clear_regions_button)
        body.addWidget(self.atlas_panel)
        self.sphere_panel = QWidget(); sphere_form = QFormLayout(self.sphere_panel); sphere_form.setContentsMargins(0, 0, 0, 0)
        self.center = [self.number(-1000, 1000, 0, ' mm') for _ in range(3)]
        coordinate_row=QWidget();coordinate_layout=QHBoxLayout(coordinate_row);coordinate_layout.setContentsMargins(0,0,0,0);coordinate_layout.setSpacing(5)
        for axis, control in zip(('X', 'Y', 'Z'), self.center):
            coordinate_layout.addWidget(QLabel(axis));coordinate_layout.addWidget(control,1)
            control.setMinimumWidth(70);control.valueChanged.connect(self.coordinates_changed)
        sphere_form.addRow(self.label('个体坐标（mm）','muted'));sphere_form.addRow(coordinate_row)
        self.crosshair_button = QPushButton('使用当前十字光标'); self.crosshair_button.clicked.connect(self.use_crosshair)
        sphere_form.addRow(self.crosshair_button)
        sphere_form.addRow(self.label('在左侧 MRI 移动十字光标后点击上方按钮，或直接填写 XYZ 坐标。', 'muted'))
        body.addWidget(self.sphere_panel)
        self.radius_row = QWidget(); radius_layout = QHBoxLayout(self.radius_row); radius_layout.setContentsMargins(0,0,0,0)
        self.radius = self.number(.5,100,10,' mm'); radius_layout.addWidget(self.label('目标区域半径')); radius_layout.addWidget(self.radius,1)
        self.radius.setToolTip('以选定位置为中心的球形目标区域。'); body.addWidget(self.radius_row)
        self.radius.valueChanged.connect(self.preview_target)
        self.shared_target_layout.addWidget(card)

        self.advanced = Collapsible('高级设置 · 电流上限与搜索参数',overlay=True)
        form = QFormLayout(self.advanced.body); self.advanced_form=form; form.setContentsMargins(12, 8, 12, 8); form.setSpacing(10)
        self.roi_mode.currentIndexChanged.connect(self.update_roi_mode)
        self.goal = QComboBox(); self.goal.addItems(['优先集中在目标区域', '优先提高目标区域场强']); self.goal.setCurrentIndex(1); form.addRow('优化方向', self.goal)
        self.cap = self.number(.1, 20, 2, ' mA'); form.addRow('每通道峰值上限', self.cap)
        self.cap.valueChanged.connect(self.sync_current_limit)
        self.minimum = self.number(.01, 20, .1, ' mA'); form.addRow('每通道峰值下限', self.minimum)
        self.tes_cap=self.number(.01,20,2,' mA',3);form.addRow('单电极电流绝对值上限',self.tes_cap)
        self.tes_total=self.number(.02,40,4,' mA',3);form.addRow('电流绝对值之和上限',self.tes_total)
        self.tes_count=self.integer(2,16,5);form.addRow('电极数量',self.tes_count)
        self.tes_cap.valueChanged.connect(self.sync_tes_current_limit)
        self.tes_total.valueChanged.connect(self.update_current_preview)
        self.tes_count.valueChanged.connect(self.rebuild_tes_manual_rows)
        self.threshold = self.number(.001, 10, .2, ' V/m', 3); form.addRow('评价场强阈值', self.threshold)
        self.weight = self.number(0, 10, 1); form.addRow('非目标区域惩罚权重', self.weight)
        from core.search_defaults import DEFAULT_SEARCH
        self.population = self.integer(8, 4096, DEFAULT_SEARCH['population']); form.addRow('搜索种群大小', self.population)
        self.generations = self.integer(1, 2000, DEFAULT_SEARCH['generations']); form.addRow('搜索代数', self.generations)
        self.seed = self.integer(0, 2147483647, 20261012); form.addRow('随机种子', self.seed)
        self.ratio_steps = self.integer(2, 1001, DEFAULT_SEARCH['ratio_steps']); form.addRow('每通道电流搜索档数', self.ratio_steps)
        self.sample_size = self.integer(64, 131072, DEFAULT_SEARCH['sample_per_region']); form.addRow('每区域搜索采样数', self.sample_size)
        self.finalists = self.integer(1, 256, DEFAULT_SEARCH['finalist_count']); form.addRow('完整复算候选数', self.finalists)
        for control,hint in ((self.population,'同时探索更多方案；增大会增加每代计算量。'),(self.generations,'继续改进更多轮；不能保证达到全局最优。'),(self.ratio_steps,'提高电流搜索精度；不改变电流安全上限。'),(self.sample_size,'提高目标与非目标区域的评估代表性。'),(self.finalists,'对更多候选用全部脑体素复算，降低采样排序的影响。')):control.setToolTip(hint)
        self.candidate_labels = QLineEdit(); self.candidate_labels.setPlaceholderText('留空使用全部电极；例如 F3,F4,P3,P4,C3,C4')
        form.addRow('候选电极', self.candidate_labels); layout.addWidget(self.advanced)
        self.tes_note=self.label('','muted');layout.addWidget(self.tes_note)
        self.optimize_button = QPushButton('自动规划并查看电场'); self.optimize_button.setObjectName('primary')
        self.optimize_button.clicked.connect(self.start_optimization)
        self.optimize_button.setToolTip('计算完成后，方案、电流和电场会自动显示并保存。')
        layout.addStretch(); self.add_page(page, '2 靶区')

    def build_manual_page(self):
        page = QWidget(); layout = QVBoxLayout(page); layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(14)
        layout.addWidget(self.label('靶区可选；已有电极方案可直接计算电场。', 'muted'))
        self.manual_target_host=QVBoxLayout();layout.addLayout(self.manual_target_host)
        self.manual_stack=QStackedWidget();self.manual_stack.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Fixed);layout.addWidget(self.manual_stack)
        ti_page=QWidget();ti_layout=QHBoxLayout(ti_page);ti_layout.setContentsMargins(0,0,0,0);ti_layout.setSpacing(8)
        names = electrode_names(self.models); self.electrodes, self.currents = [], []
        for k, defaults in enumerate((('F3', 'F4'), ('P3', 'P4'))):
            card, body = self.card(f'通道 {k+1}'); form = QFormLayout()
            for title, default in zip(('正极', '负极'), defaults):
                combo = QComboBox()
                for label, name in names.items(): combo.addItem(name, label)
                combo.setCurrentIndex(max(combo.findText(default), 0)); form.addRow(title, combo); self.electrodes.append(combo)
            current = self.number(.01, self.cap.value(), 2, ' mA', 3); form.addRow('峰值', current)
            self.currents.append(current); body.addLayout(form); ti_layout.addWidget(card,1)
        self.manual_stack.addWidget(ti_page)
        tes_page=QWidget();tes_layout=QVBoxLayout(tes_page);tes_layout.setContentsMargins(0,0,0,0)
        tes_layout.addWidget(self.label('为每个电极填写带符号电流。正负电流之和必须为 0；直流值或交流峰值，单位 mA。','muted'))
        self.tes_rows=QWidget();self.tes_rows_layout=QFormLayout(self.tes_rows);tes_layout.addWidget(self.tes_rows)
        self.tes_electrodes=[];self.tes_currents=[];self.rebuild_tes_manual_rows()
        self.manual_stack.addWidget(tes_page)
        for control in self.currents: control.valueChanged.connect(self.update_current_preview)
        self.current_preview_label=self.label('');layout.addWidget(self.current_preview_label)
        self.synthesize_button = QPushButton('计算并查看电场'); self.synthesize_button.setObjectName('primary')
        self.synthesize_button.clicked.connect(self.start_synthesis)
        layout.addStretch(); self.add_page(page, '手动方案')

    def resize_manual_controls(self):
        current=self.manual_stack.currentWidget()
        if current is not None:self.manual_stack.setFixedHeight(current.sizeHint().height()+2)

    def show_shared_target_controls(self, index):
        host=self.manual_target_host if index==2 else self.optimization_target_host
        if host.indexOf(self.shared_target_controls)<0:
            host.addWidget(self.shared_target_controls)
        self.shared_target_controls.show()
        QTimer.singleShot(0,self.resize_manual_controls)

    def preview_target(self, *args):
        if not self.task or not self.task.get('padded_t1') or Path(self.owner.current_base_path or '').resolve()!=Path(self.task['padded_t1']).resolve():return
        try:roi=self.roi_spec()
        except ValueError:roi=None
        self.owner.set_target_roi(roi)

    def sync_current_limit(self, *_signal_args):
        for control in getattr(self,'currents',[]):control.setMaximum(self.cap.value())

    def is_tes(self):
        return self.stimulation_mode.currentIndex()==1

    def update_stimulation_mode(self,*args):
        tes=self.is_tes()
        for control in (self.cap,self.minimum):
            control.setVisible(not tes);self.advanced_form.labelForField(control).setVisible(not tes)
        for control in (self.tes_cap,self.tes_total,self.tes_count):
            control.setVisible(tes);self.advanced_form.labelForField(control).setVisible(tes)
        self.advanced_form.labelForField(self.ratio_steps).setText('单电极电流搜索档数' if tes else '每通道电流搜索档数')
        self.manual_stack.setCurrentIndex(1 if tes else 0)
        QTimer.singleShot(0,self.resize_manual_controls)
        self.tes_note.setVisible(tes)
        self.stimulation_mode.setToolTip('tES 共用直流和同频交流电场模型；交流按 0°/180°相位及峰值电流计算。')
        self.update_current_preview()

    def rebuild_tes_manual_rows(self,*args):
        if not hasattr(self,'tes_rows_layout'):return
        saved=[(combo.currentData(),current.value()) for combo,current in zip(self.tes_electrodes,self.tes_currents)]
        while self.tes_rows_layout.rowCount():self.tes_rows_layout.removeRow(0)
        self.tes_electrodes=[];self.tes_currents=[]
        names=electrode_names(self.models);defaults=['C3','C1','C5','FC3','CP3'];count=self.tes_count.value()
        used=set()
        for i in range(count):
            row=QWidget();row_layout=QHBoxLayout(row);row_layout.setContentsMargins(0,0,0,0)
            combo=QComboBox()
            for label,name in names.items():combo.addItem(name,label)
            if i<len(saved):combo.setCurrentIndex(combo.findData(saved[i][0]));value=saved[i][1]
            else:
                desired=combo.findText(defaults[i]) if i<len(defaults) else -1
                if desired<0 or combo.itemData(desired) in used:
                    desired=next(j for j in range(combo.count()) if combo.itemData(j) not in used)
                combo.setCurrentIndex(desired)
                value=2. if i==0 else -2./(count-1)
            used.add(combo.currentData())
            current=self.number(-self.tes_cap.value(),self.tes_cap.value(),value,' mA',8)
            current.valueChanged.connect(self.update_current_preview)
            row_layout.addWidget(combo,1);row_layout.addWidget(current,1)
            self.tes_rows_layout.addRow(f'电极 {i+1}',row)
            self.tes_electrodes.append(combo);self.tes_currents.append(current)
        self.update_current_preview()
        QTimer.singleShot(0,self.resize_manual_controls)

    def sync_tes_current_limit(self,*args):
        for control in getattr(self,'tes_currents',[]):control.setRange(-self.tes_cap.value(),self.tes_cap.value())
        self.update_current_preview()

    def locate_target(self, *_signal_args):
        try:
            if not self.task:raise ValueError('请先生成个体模型。')
            self.owner.locate_roi(self.roi_spec(),self.task['padded_t1'])
        except Exception as exc:self.show_error('请先选择靶区',str(exc))

    def browse_t1(self, *_signal_args):
        path, _ = QFileDialog.getOpenFileName(self, '选择 T1 MRI', self.t1.text(), 'MRI 影像 (*.nii *.nii.gz)')
        if path:
            self.select_mri(path)

    def import_dicom(self, *_signal_args):
        if self.process is not None:
            return
        from ui.dicom_import import DicomImportDialog
        dialog = DicomImportDialog(self.storage_root.parent / 'dicom_imports', self)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.selected_path:
            if self.select_mri(dialog.selected_path):
                self.source_note.setText('已载入医生确认的 DICOM 转换影像。请检查患者、影像内容及方向，再点击“生成电场模型”。')

    def use_builtin_mri(self, *_signal_args):
        if self.process is not None:
            return
        try:
            self.select_mri(str(builtin_mri_path()))
        except FileNotFoundError as exc:
            self.show_error('内置 MRI 不可用', str(exc))

    def select_mri(self, path):
        if self.process is not None:
            return False
        path = str(path)
        if not self.owner.execute_base_mri_load(path):
            return False
        self.timings.reset()
        self.task=None; self.atlas_record=None; self.target_selected=False; self.stop_target_selection()
        self.populate_regions(); self.owner.individual_task=None
        self.t1.setText(path)
        self.target_label.setText('尚未选择目标位置')
        self.task_label.setText(f'已选择：{Path(path).name}。点击“生成电场模型”即可开始。')
        self.tabs.setCurrentIndex(0); self.set_busy(False); self.update_roi_mode()
        self.check_existing_model()
        return True

    def update_mri_selection(self,value):
        self.t1_name.setText(Path(value).name if value else '尚未选择 MRI 文件'); self.t1_name.setToolTip(value)
        if hasattr(self, 'source_note'):
            try:
                reference = bool(value) and Path(value).resolve() == builtin_mri_path().resolve()
            except FileNotFoundError:
                reference = False
            self.source_note.setText('当前使用内置参考 MRI：用于标准头模型仿真，不代表患者个人解剖。' if reference
                                     else '支持 T1 MRI（.nii / .nii.gz），或导入 DICOM 并确认 T1 序列。没有 MRI 时可使用内置影像。')
        if hasattr(self,'predict_button'): self.predict_button.setEnabled(self.process is None and Path(value).is_file())

    def set_busy(self, busy):
        self.stimulation_mode.setEnabled(not busy)
        for section in (self.advanced,):
            section.body.setEnabled(not busy)
            if busy:section.set_expanded(False)
        self.tabs.setEnabled(not busy)
        self.tabs.setTabEnabled(1, self.task is not None); self.tabs.setTabEnabled(2, self.task is not None)
        self.predict_button.setEnabled(not busy and Path(self.t1.text()).is_file())
        self.synthesize_button.setEnabled(not busy and self.task is not None)
        self.optimize_button.setEnabled(not busy and self.task is not None); self.cancel_button.setEnabled(busy)
        self.atlas_prepare.setEnabled(not busy and self.task is not None)
        self.locate_button.setEnabled(not busy and self.task is not None)
        self.cancel_button.setVisible(busy)
        self.browse_button.setEnabled(not busy); self.builtin_button.setEnabled(not busy)
        self.dicom_button.setEnabled(not busy)
        self.progress.setVisible(busy)
        if busy: self.progress.setRange(0, 0)

    def update_current_preview(self, *_signal_args):
        if not hasattr(self,'current_preview_label'):return
        if self.is_tes():
            self.tes_note.setText(f'使用 {self.tes_count.value()} 个电极；单电极 ≤{self.tes_cap.value():g} mA，电流绝对值合计 ≤{self.tes_total.value():g} mA。')
            values=[c.value() for c in self.tes_currents]
            total=sum(abs(v) for v in values);balance=sum(values)
            if abs(balance)<1e-8:balance=0.
            self.current_preview_label.setText(f'tES · {len(values)} 电极 · 绝对值合计 {total:.5f} mA · 带符号合计 {balance:+.5f} mA'+('（请调整为 0）' if abs(balance)>1e-8 else '（已平衡）'))
            return
        self.current_preview_label.setText(f'通道1 ±{self.currents[0].value():g} mA   ·   通道2 ±{self.currents[1].value():g} mA（峰值）')

    def update_roi_mode(self, *_signal_args):
        point=self.roi_mode.currentIndex()==0
        if not point:self.stop_target_selection()
        self.sphere_panel.setVisible(point)
        self.radius_row.setVisible(point)
        self.atlas_panel.setVisible(not point)
        self.sphere_panel.setEnabled(point and self.task is not None);self.radius.setEnabled(point)
        self.preview_target()

    def coordinates_changed(self, *_signal_args):
        if self.task:
            self.target_selected = True; self.target_label.setText('目标位置已设置，可定位、优化或计算手动方案。')
            self.preview_target()

    def begin_target_selection(self):
        if not self.task: return
        self.roi_mode.setCurrentIndex(0)
        if Path(self.owner.current_base_path or '').resolve() != Path(self.task['padded_t1']).resolve(): self.task_ready.emit(self.task)
        self.selection_mode = True; self.owner.stack.setCurrentIndex(1)
        self.owner.statusBar().showMessage('请在脑部切片上点击希望作用的位置。', 0)
        self.status.setText('请在左侧脑部切片上点击目标位置。')

    def stop_target_selection(self, *args):
        if self.selection_mode:
            self.selection_mode = False
            self.owner.statusBar().clearMessage()
            self.status.setText('已取消选点，可以重新选择目标。')

    def use_crosshair(self, *_signal_args):
        if not self.task or Path(self.owner.current_base_path or '').resolve() != Path(self.task['padded_t1']).resolve():
            self.show_error('先加载您的个体模型', '当前查看的影像与个体模型不一致。'); return
        xyz = self.owner.current_cursor_ras()
        for control, value in zip(self.center, xyz): control.setValue(value)
        self.target_selected = True; self.target_label.setText('已选目标：' + '，'.join(f'{a} {float(v):.1f}' for a,v in zip(('X','Y','Z'),xyz)) + ' mm')
        self.preview_target()

    def capture_target_from_viewer(self):
        if not self.selection_mode: return
        self.use_crosshair(); self.selection_mode = False; self.owner.statusBar().clearMessage()
        self.status.setText('目标已记录，确认范围和电流后即可自动规划。')
        self.tabs.setCurrentIndex(1)

    def roi_spec(self):
        if self.roi_mode.currentIndex()==1:
            if not self.atlas_record: raise ValueError('请先准备 AAL3 个体脑区图谱。')
            labels=self.selected_regions()
            if not labels: raise ValueError('请先搜索并勾选希望作用的脑区。')
            return {'mask':self.atlas_record['atlas_path'],'labels':labels,'atlas_id':'aal3',
                'atlas_version':self.atlas_record['atlas_version'],
                'region_names':[self.atlas_record['labels'][str(k)] for k in labels]}
        if self.roi_mode.currentIndex() == 0:
            if not self.target_selected: raise ValueError('请填写个体 XYZ 坐标，或使用当前十字光标设置目标位置。')
            return {'center_ras_mm': [v.value() for v in self.center], 'radius_mm': self.radius.value()}
        raise ValueError('请选择靶区。')

    @staticmethod
    def task_from_manifest(path):
        path = Path(path).resolve(); m = json.loads(path.read_text(encoding='utf-8')); stages = m['stages']
        return {'task_dir': str(path.parent), 'manifest': str(path), 'padded_t1': stages['padded'], 'prediction_dir': str(path.parent/'prediction'),
                'tissue_masks': stages['segmentation'], 'electrodes': stages['electrodes'], 't1': m.get('t1', ''), 'atlas':m.get('atlases',{}).get('aal3'), 'timings':m.get('timings',stages.get('preprocessing',{}).get('timings',{}))}

    def attach_task(self, task):
        from core.ti import Leadfield
        lf = Leadfield(task['prediction_dir']); lf.close()
        if not Path(task['padded_t1']).is_file(): raise ValueError('保存的模型影像不存在，请重新生成模型。')
        changed = self.task is None or self.task['task_dir'] != task['task_dir']; self.task = task
        if changed:
            self.target_selected = False; self.stop_target_selection(); self.target_label.setText('尚未选择目标位置')
        if task.get('t1'): self.t1.setText(task['t1'])
        self.atlas_record=task.get('atlas')
        if self.atlas_record and not Path(self.atlas_record['atlas_path']).is_file(): self.atlas_record=None
        self.populate_regions()
        self.task_label.setText('模型已准备好。接下来选择靶区，或使用手动电极方案。')
        if task.get('timings') and not self.timings.records:
            self.timings.restore({key:dict(value,state='cached') for key,value in task['timings'].items()})
        self.task_ready.emit(task); self.set_busy(False); self.update_roi_mode(); self.status.setText('模型已就绪，下一步选择靶区。')

    def start_atlas(self, *_signal_args):
        if not self.task: return
        self.run({'action':'atlas','task_dir':self.task['task_dir'],'padded_t1':self.task['padded_t1'],
            'tissue_masks':self.task['tissue_masks'],'atlas_resources':str(self.atlas_resources)})

    def selected_regions(self):
        return [self.region_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.region_list.count())
            if self.region_list.item(i).checkState()==Qt.CheckState.Checked]

    def populate_regions(self):
        self.region_list.blockSignals(True); self.region_list.clear()
        if self.atlas_record:
            from core.atlas import region_name
            for key,name in self.atlas_record['labels'].items():
                item=QListWidgetItem(region_name(name)+'  '+name); item.setData(Qt.ItemDataRole.UserRole,int(key))
                item.setFlags(item.flags()|Qt.ItemFlag.ItemIsUserCheckable); item.setCheckState(Qt.CheckState.Unchecked)
                if self.atlas_record.get('gmwm_counts',{}).get(key,1)==0:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled); item.setToolTip('当前导联场未覆盖此脑区')
                self.region_list.addItem(item)
        self.region_list.blockSignals(False)
        ready=self.atlas_record is not None
        self.region_search.setEnabled(ready); self.region_list.setEnabled(ready); self.clear_regions_button.setEnabled(ready)
        self.region_list.setVisible(ready); self.region_search.setVisible(ready); self.clear_regions_button.setVisible(ready)
        self.atlas_prepare.setVisible(not ready)
        self.atlas_status.setText('AAL3 已映射到您的 MRI，可选择左侧、右侧或多个脑区。' if ready else '首次准备需要一些时间，同一模型会自动复用。')
        self.filter_regions()

    def filter_regions(self, *_signal_args):
        query=self.region_search.text().strip().casefold()
        for i in range(self.region_list.count()):
            item=self.region_list.item(i); item.setHidden(query not in item.text().casefold())

    def regions_changed(self, *args):
        if not self.atlas_record: return
        from core.atlas import region_name
        labels=self.selected_regions(); names=[region_name(self.atlas_record['labels'][str(k)]) for k in labels]
        self.target_label.setText('已选靶区：'+'、'.join(names) if labels else '搜索并勾选希望作用的脑区')
        if self.roi_mode.currentIndex()==1:self.owner.preview_atlas_regions(self.atlas_record,labels)

    def clear_regions(self, *_signal_args):
        self.region_list.blockSignals(True)
        for i in range(self.region_list.count()): self.region_list.item(i).setCheckState(Qt.CheckState.Unchecked)
        self.region_list.blockSignals(False); self.regions_changed()

    def check_existing_model(self):
        self._inspected_t1 = self.t1.text()
        self.run({'action':'inspect', 't1':self._inspected_t1, 'brain_root':str(self.models),
                  'output_root':str(self.storage_root), 'padding':10})

    def offer_existing_model(self, result):
        if self.t1.text() != self._inspected_t1: return
        task = result.get('task')
        if not task:
            self.status.setText('影像已加载，可以生成个体模型。')
            self.task_label.setText('未找到可复用的完整模型。点击上方按钮即可生成。')
            return
        prompt = QMessageBox(self); prompt.setWindowTitle('发现已有个体模型')
        prompt.setIcon(QMessageBox.Icon.Question)
        prompt.setText('这张 MRI 已有完整的导联场，是否直接使用？')
        prompt.setInformativeText('MRI、模型权重和处理参数已核对。使用后可直接选择靶区或计算电场。')
        use = prompt.addButton('使用已有模型', QMessageBox.ButtonRole.AcceptRole)
        prompt.addButton('暂不使用', QMessageBox.ButtonRole.RejectRole)
        prompt.setDefaultButton(use); prompt.exec()
        if prompt.clickedButton() is use:
            self.attach_task(task); self.tabs.setCurrentIndex(1)
            self.status.setText('已使用已有个体模型，请选择靶区。')
        else:
            self.status.setText('影像已加载，已有模型暂未使用。')

    def start_prediction(self, *_signal_args):
        if not Path(self.t1.text()).is_file(): self.show_error('请先选择 MRI', '支持 .nii 或 .nii.gz 格式的 T1 影像。'); return
        self.run({'action':'predict', 't1':self.t1.text(), 'brain_root':str(self.models), 'output_root':str(self.storage_root), 'padding':10, 'device':'auto'})

    def job_config(self, action):
        if not self.task: raise ValueError('请先生成或选择已保存的个体模型。')
        name = datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid4().hex[:6]
        config={'action':action, 'prediction_dir':self.task['prediction_dir'], 'output_dir':str(Path(self.task['task_dir'])/action/name)}
        if self.is_tes():config['stimulation_type']='tES'
        return config

    def start_synthesis(self, *_signal_args):
        try:
            config = self.job_config('synthesize')
            if self.is_tes():
                config.update(electrode_labels=[x.currentData() for x in self.tes_electrodes],currents=[x.value() for x in self.tes_currents],
                    constraints={'electrode_limit_ma':self.tes_cap.value(),'absolute_sum_limit_ma':self.tes_total.value(),'electrode_count':self.tes_count.value()})
            else:
                config.update(pairs=[[x.currentData() for x in self.electrodes[:2]], [x.currentData() for x in self.electrodes[2:]]], currents=[x.value() for x in self.currents])
            try:config['roi']=self.roi_spec()
            except ValueError:pass  # A manual montage can also be evaluated before choosing a target.
            from core.ti import Leadfield, validate_montage
            lf = Leadfield(config['prediction_dir'])
            try:
                if self.is_tes():
                    from core.tes import validate_montage as validate_tes
                    validate_tes(lf,config['electrode_labels'],config['currents'],**config['constraints'])
                else:validate_montage(lf, config['pairs'], config['currents'])
            finally: lf.close()
            self.run(config)
        except Exception as exc: self.show_error('请检查电极方案', str(exc))

    def start_optimization(self, *_signal_args):
        try:
            config = self.job_config('optimize'); config['roi'] = self.roi_spec()
            from core.ti import Leadfield
            lf = Leadfield(config['prediction_dir'])
            try: lf.roi(config['roi'])
            finally: lf.close()
            options = {'channel_limit_ma':self.cap.value(), 'min_channel_ma':self.minimum.value(),
                'threshold_v_m':self.threshold.value(), 'avoidance_weight':self.weight.value(), 'population':self.population.value(),
                'generations':self.generations.value(), 'seed':self.seed.value(), 'ratio_steps':self.ratio_steps.value(),
                'sample_per_region':self.sample_size.value(), 'finalist_count':self.finalists.value(), 'goal':'focality' if self.goal.currentIndex()==0 else 'intensity'}
            if self.is_tes():
                options.pop('channel_limit_ma');options.pop('min_channel_ma')
                options.update(electrode_limit_ma=self.tes_cap.value(),absolute_sum_limit_ma=self.tes_total.value(),electrode_count=self.tes_count.value())
            elif options['min_channel_ma']>options['channel_limit_ma']:
                raise ValueError('每通道电流下限不能高于上限，请检查高级设置。')
            if self.candidate_labels.text().strip():
                lookup = {name.casefold():label for label,name in electrode_names(self.models).items()}
                options['electrodes'] = [int(x) if x.strip().isdigit() else lookup[x.strip().casefold()] for x in self.candidate_labels.text().split(',')]
            config['options'] = options; self.run(config)
        except Exception as exc: self.show_error('还不能开始自动规划', str(exc))

    def run(self, config):
        if self.process is not None: return
        try:
            request_dir = self.storage_root/'requests'; request_dir.mkdir(parents=True, exist_ok=True)
            key = uuid4().hex; path = request_dir/f'{key}.json'; path.write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
            logs = self.storage_root/'logs'; logs.mkdir(exist_ok=True); self._log_stream = (logs/f'{key}.log').open('w',encoding='utf-8')
            self._result = self._error = None; self._buffer = ''; self._cancelled = False
            if config['action']=='predict': self.timings.reset()
            self._decoder = codecs.getincrementaldecoder('utf-8')(errors='replace'); self._action = config['action']
            self.process = QProcess(self); self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
            self.process.readyReadStandardOutput.connect(self.read_output); self.process.finished.connect(self.finished); self.process.errorOccurred.connect(self.process_error)
            executable,args = worker_command(self.root,path)
            self.stop_target_selection(); self.set_busy(True); self.status.setText('正在检查已有个体模型…' if self._action=='inspect' else '正在准备计算，请稍候…'); self.process.start(executable,args)
        except Exception as exc:
            self.process = None
            if self._log_stream: self._log_stream.close(); self._log_stream = None
            self.set_busy(False); self.show_error('暂时无法启动计算',str(exc))

    def display_progress(self, message):
        import re
        if 'CPU' in message and '导联场' in message and 'leadfield_prediction' in self.timings.records:
            self.timings.records['leadfield_prediction']['device']='cpu'; self.timings.render()
        if 'BrainLFNet 预测' in message or '复用导联场列' in message:
            match = re.search(r'(\d+)/71',message)
            if match:
                done=int(match.group(1)); done = max(0, done-1) if '预测' in message else done
                self.progress.setRange(0,71); self.progress.setValue(done); self.status.setText(f'正在完成个体模型 · {round(100*done/71)}%'); return
        if '遗传搜索' in message:
            match = re.search(r'(\d+)/(\d+)',message)
            if match:
                done,total=map(int,match.groups()); self.progress.setRange(0,total); self.progress.setValue(done); self.status.setText(f'正在寻找合适方案 · {round(100*done/total)}%'); return
        for token,text in [('AAL3','正在准备个体脑区图谱'),('核对','正在准备 MRI 和模型'),('统一 MRI','正在整理 MRI 影像'),('nnU-Net','正在识别头部组织'),('组织分割完成','影像识别完成'),('清理组织','正在整理头部组织'),
                           ('电极放置','正在计算个体电极位置'),('Python affine','正在定位头部标志点'),('完整导联场复算','正在核对候选方案'),('TI 合成','正在生成电场图'),('tES 电场合成','正在生成电场图'),('复用已完成','正在读取已保存的个体模型')]:
            if token in message:
                if token == '完整导联场复算': self.progress.setRange(0, 0)
                self.status.setText(text+'…'); return
        if '显卡' in message or 'CPU' in message:
            self.status.setText(message)

    @staticmethod
    def irrelevant_nnunet_message(line):
        return any(line.strip().startswith(name+' is not defined and nnU-Net')
                   for name in ('nnUNet_raw','nnUNet_preprocessed','nnUNet_results'))

    def read_output(self):
        if self.process is None: return
        decoded = self._decoder.decode(bytes(self.process.readAllStandardOutput()))
        if self._log_stream: self._log_stream.write(decoded); self._log_stream.flush()
        self._buffer += decoded; lines=self._buffer.split('\n'); self._buffer=lines.pop()
        for line in lines:
            if line.startswith('NEUROGUIDE_EVENT '):
                try:
                    event=json.loads(line[len('NEUROGUIDE_EVENT '):])
                    if event['kind']=='result': self._result=event['value']
                    elif event['kind']=='timing': self.timings.update_stage(event['value'])
                    elif event['kind']=='error': self._error=event['value']
                    else: self.display_progress(event['value']); self.log.appendPlainText(event['value'])
                except (ValueError,KeyError): self.log.appendPlainText(line)
            elif line.strip() and not self.irrelevant_nnunet_message(line): self.log.appendPlainText(line.rstrip())

    def process_error(self,error):
        if error==QProcess.ProcessError.FailedToStart:
            self._error=self.process.errorString(); self.finished(1,QProcess.ExitStatus.CrashExit)

    def finished(self,code,status):
        if self.process is None: return
        self.read_output(); self.process.deleteLater(); self.process=None
        if self._log_stream: self._log_stream.close(); self._log_stream=None
        self.set_busy(False); self.timings.stopped()
        if self._cancelled: self.status.setText('已取消。下次继续时，会自动复用已完成的部分。')
        elif (code or self._result is None) and self._action=='inspect':
            self.status.setText('已有模型检查未完成，仍可生成个体模型。')
            self.task_label.setText('暂时无法核对已有模型。可点击上方按钮生成；详情见运行日志。')
        elif code or self._result is None:
            self.status.setText('本次计算未完成，可展开运行详情查看原因。'); self.show_error('计算未完成',self._error or '后台计算没有返回完整结果。')
        else:
            try:
                if self._action=='inspect': self.offer_existing_model(self._result)
                elif self._action=='predict': self.attach_task(self._result); self.tabs.setCurrentIndex(1)
                elif self._action=='atlas':
                    self.atlas_record=self._result; self.task['atlas']=self._result; self.populate_regions()
                    self.tabs.setCurrentIndex(1); self.atlas_ready.emit(self._result); self.status.setText('脑区图谱已就绪，请搜索并勾选靶区。')
                else: self.apply_result(self._result); self.field_ready.emit(self._result); self.status.setText('方案已保存，电场已显示。')
            except Exception as exc: self.show_error('结果暂时无法显示',str(exc))

    def apply_result(self,result):
        tes=result.get('stimulation_type')=='tES';self.stimulation_mode.setCurrentIndex(1 if tes else 0)
        if tes:
            limits=result.get('constraints',{})
            self.tes_cap.setValue(max(limits.get('electrode_limit_ma',2.),max(map(abs,result['currents_ma']))))
            self.tes_total.setValue(limits.get('absolute_sum_limit_ma',4.))
            self.tes_count.setValue(len(result['electrode_labels']))
            for combo,label in zip(self.tes_electrodes,result['electrode_labels']):combo.setCurrentIndex(combo.findData(label))
            displayed=[round(float(c),8) for c in result['currents_ma']]
            residual=sum(displayed)
            if abs(residual)>1e-12:
                # Preserve current balance after rounding to the spinbox's
                # precision, using an electrode with room below the cap.
                for i,value in enumerate(displayed):
                    corrected=round(value-residual,8)
                    if abs(corrected)<=self.tes_cap.value() and abs(corrected)>1e-9:
                        displayed[i]=corrected;break
            for control,value in zip(self.tes_currents,displayed):control.setValue(value)
        else:
            self.cap.setValue(max(self.cap.value(),max(result['currents_peak_ma'])))
            for combo,label in zip(self.electrodes,[x for p in result['pairs'] for x in p]): combo.setCurrentIndex(combo.findData(label))
            for control,value in zip(self.currents,result['currents_peak_ma']): control.setValue(value)
        roi = result.get('roi') or {}
        if 'center_ras_mm' in roi:
            self.roi_mode.setCurrentIndex(0)
            for control,value in zip(self.center,roi['center_ras_mm']): control.setValue(value)
            self.radius.setValue(roi['radius_mm']); self.target_selected = True
            self.target_label.setText('已选目标：' + '，'.join(f'{a} {float(v):.1f}' for a,v in zip(('X','Y','Z'),roi['center_ras_mm'])) + ' mm')
            self.stop_target_selection()
        elif roi.get('mask') and roi.get('atlas_id')=='aal3' and self.atlas_record:
            self.roi_mode.setCurrentIndex(1); self.region_list.blockSignals(True)
            for i in range(self.region_list.count()):
                item=self.region_list.item(i); item.setCheckState(Qt.CheckState.Checked if item.data(Qt.ItemDataRole.UserRole) in roi.get('labels',[]) else Qt.CheckState.Unchecked)
            self.region_list.blockSignals(False)
            from core.atlas import region_name
            self.target_label.setText('已选靶区：'+'、'.join(region_name(self.atlas_record['labels'][str(k)]) for k in self.selected_regions()))
        elif roi.get('mask'):
            self.roi_mode.setCurrentIndex(1)
            self.target_label.setText('已载入历史目标区域；重新规划时请点选或选择 AAL3 脑区。')
        self.log.appendPlainText(current_description(result,electrode_names(self.models)))
        stats=result.get('statistics',{})
        if 'roi' in stats: self.log.appendPlainText(f"目标区域均值 {stats['roi']['mean_v_m']:.4f} V/m，峰值 {stats['roi']['max_v_m']:.4f} V/m")

    def show_error(self,title,details):
        box=QMessageBox(self); box.setIcon(QMessageBox.Icon.Warning); box.setWindowTitle(title); box.setText(title)
        box.setInformativeText(str(details).splitlines()[0][:180]); box.setDetailedText(str(details)); box.exec()

    def cancel(self, *_signal_args):
        if self.process is None: return
        self._cancelled=True; self.status.setText('正在取消计算…'); pid=int(self.process.processId())
        if os.name=='nt' and pid:
            import subprocess
            subprocess.run(['taskkill','/PID',str(pid),'/T','/F'],capture_output=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        else: self.process.kill()
