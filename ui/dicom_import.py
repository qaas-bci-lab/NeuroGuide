"""Doctor-confirmed DICOM series selection; no automatic T1 classification."""
from pathlib import Path

from PyQt6.QtCore import Qt, QThread
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QDialog, QFileDialog,
    QHBoxLayout, QHeaderView, QInputDialog, QLabel, QMessageBox, QProgressBar,
    QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout)

from core.dicom_import import scan_series, preview_series, convert_series


class ImportJob(QThread):
    def __init__(self, operation, parent):
        super().__init__(parent)
        self.operation = operation
        self.result = None
        self.error = None

    def run(self):
        try:
            self.result = self.operation(self.isInterruptionRequested)
        except Exception as exc:
            self.error = str(exc)


class DicomImportDialog(QDialog):
    def __init__(self, output_root, parent=None):
        super().__init__(parent)
        self.output_root = Path(output_root)
        self.selected_path = None
        self.records = []
        self.job = None
        self.closing = False
        self.setWindowTitle('导入 DICOM · 确认 T1 序列')
        self.resize(1060, 720)
        layout = QVBoxLayout(self)
        note = QLabel('先选择文件夹，再核对患者、检查日期和序列。名称不能保证影像为 T1，请结合原始检查信息与预览确认。')
        note.setWordWrap(True); layout.addWidget(note)
        row = QHBoxLayout()
        self.folder_label = QLabel('尚未选择 DICOM 文件夹'); row.addWidget(self.folder_label, 1)
        self.browse = QPushButton('选择 DICOM 文件夹…'); self.browse.clicked.connect(self.choose_folder); row.addWidget(self.browse)
        layout.addLayout(row)
        self.table = QTableWidget(0, 9)
        self.table.setStyleSheet('QTableWidget { background: #171a21; color: #e1e5ed; gridline-color: #353943; border: 1px solid #353943; } '
            'QTableWidget::item:selected { background: #2a4165; color: white; } '
            'QHeaderView::section { background: #282c35; color: #e1e5ed; border: 1px solid #353943; padding: 5px; } '
            'QTableCornerButton::section { background: #282c35; border: 1px solid #353943; }')
        self.table.setHorizontalHeaderLabels(['患者姓名', '患者编号', '检查日期', '序列号', '序列描述 / 协议', '模态', '文件数', '矩阵 / 帧数', '图像类型'])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self.selection_changed)
        layout.addWidget(self.table, 1)
        self.preview = QLabel('选择序列后显示样本预览（不代表完整体积，也不作 T1 自动判定）')
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter); self.preview.setMinimumHeight(180)
        layout.addWidget(self.preview)
        caption = QLabel('样本预览：不代表完整体积，未经解剖方向重排；不用于核对左右方向，也不作 T1 自动判定。')
        caption.setWordWrap(True); layout.addWidget(caption)
        self.confirm = QCheckBox('我已核对患者及检查信息，并确认所选序列为用于建模的 T1 结构像')
        self.confirm.toggled.connect(self.update_actions); layout.addWidget(self.confirm)
        self.status = QLabel('软件不会自动选择序列，也不会自动开始建模。'); self.status.setWordWrap(True); layout.addWidget(self.status)
        self.progress = QProgressBar(); self.progress.setRange(0, 0); self.progress.hide(); layout.addWidget(self.progress)
        actions = QHBoxLayout(); actions.addStretch()
        self.close_button = QPushButton('取消'); self.close_button.clicked.connect(self.reject); actions.addWidget(self.close_button)
        self.convert = QPushButton('转换并载入所选序列'); self.convert.setEnabled(False)
        self.convert.setObjectName('primary')
        self.convert.clicked.connect(self.start_conversion); actions.addWidget(self.convert); layout.addLayout(actions)

    def selected_record(self):
        rows = self.table.selectionModel().selectedRows()
        return self.records[rows[0].row()] if rows else None

    def update_actions(self, *_args):
        record = self.selected_record()
        self.convert.setEnabled(self.job is None and self.confirm.isChecked() and record is not None and record['modality'] == 'MR')

    def start_job(self, operation, callback, message):
        self.status.setText(message); self.progress.show()
        self.browse.setEnabled(False); self.table.setEnabled(False); self.confirm.setEnabled(False)
        self.job = ImportJob(operation, self)
        self.job.finished.connect(lambda: self.finish_job(callback))
        self.update_actions(); self.job.start()

    def finish_job(self, callback):
        job = self.job; self.job = None
        self.progress.hide(); self.browse.setEnabled(True); self.table.setEnabled(True); self.confirm.setEnabled(True)
        job.deleteLater()
        if self.closing:
            super().reject(); return
        if job.error:
            self.status.setText(job.error)
        else:
            try:
                callback(job.result)
            except Exception as exc:
                self.status.setText(str(exc))
        self.update_actions()

    def choose_folder(self, *_args):
        folder = QFileDialog.getExistingDirectory(self, '选择 DICOM 文件夹')
        if not folder:
            return
        self.confirm.setChecked(False)
        self.table.clearSelection(); self.table.setRowCount(0); self.records = []
        self.preview.setText('正在扫描…'); self.folder_label.setText(folder)
        self.start_job(lambda cancel: scan_series(folder, cancel), self.show_series, '正在扫描文件夹及子文件夹中的影像序列…')

    def show_series(self, result):
        self.records = result['series']; self.table.setRowCount(len(self.records))
        for row, r in enumerate(self.records):
            values = [r['patient'], r['patient_id'], r['date'], r['number'],
                r['description'] + ' / ' + r['protocol'], r['modality'], str(len(r['files'])),
                r['matrix'] + ' / ' + r['frames'], r['image_type']]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value); item.setToolTip(value + '\nSeries UID: ' + r['uid'])
                self.table.setItem(row, col, item)
        self.table.clearSelection(); self.confirm.setChecked(False)
        self.preview.setText('请选择并核对一个序列。预览仅为样本，不作 T1 自动判定。')
        self.status.setText(f'找到 {len(self.records)} 个序列。请核对患者、检查日期与序列信息。' if self.records else '未找到可识别的 DICOM 序列，请选择已解压的影像文件夹。')
        if result['unreadable']:
            self.status.setText(self.status.text() + f" 有 {result['unreadable']} 个序列无法读取完整信息。")

    def selection_changed(self):
        self.confirm.setChecked(False)
        record = self.selected_record()
        if record is None or self.job is not None:
            self.update_actions(); return
        self.preview.setText('正在生成样本预览…')
        self.start_job(lambda cancel: preview_series(record, cancel), self.show_preview,
            '预览仅辅助核对；请确认所选影像为 T1 结构像。' if record['modality'] == 'MR' else '该序列不是 MR，不能用于当前 T1 建模。')

    def show_preview(self, pixels):
        height, width = pixels.shape
        image = QImage(pixels.data, width, height, pixels.strides[0], QImage.Format.Format_Grayscale8).copy()
        self.preview.setPixmap(QPixmap.fromImage(image).scaled(420, 220, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.preview.setToolTip('DICOM 样本预览；未按解剖方向重排，不用于判断左右方向。转换后请在主界面检查完整影像。')
        record = self.selected_record()
        self.status.setText('请核对患者、检查信息与序列；确认是 T1 后勾选下方确认框。' if record and record['modality'] == 'MR'
                            else '该序列不是 MR，不能用于当前 T1 建模。')

    def start_conversion(self, *_args):
        record = self.selected_record()
        if self.job is not None or record is None or not self.confirm.isChecked():
            return
        self.start_job(lambda cancel: convert_series(record, self.output_root, cancel), self.conversion_done,
            '正在转换已确认的序列。原始 DICOM 不会被修改；完成后请检查完整影像。')

    def conversion_done(self, result):
        if result.get('warnings'):
            answer = QMessageBox.question(self, '转换工具提示：请核查影像',
                '转换工具报告以下问题：\n\n' + '\n'.join(result['warnings']) +
                '\n\n请核查原始影像与转换结果。是否仍要载入以检查影像？载入不会开始建模。',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel)
            if answer != QMessageBox.StandardButton.Yes:
                self.status.setText('未载入影像。转换结果及日志保存在：' + result['output']); return
        candidates = result['candidates']
        if len(candidates) > 1 or result['rejected']:
            choices = [Path(c['path']).name + '  ' + ' × '.join(map(str, c['shape'])) for c in candidates]
            label = '所选序列产生多份影像，请确认要载入的 T1：'
            if result['rejected']:
                label += '\n部分输出不符合三维导入要求，已排除；请核对剩余影像。'
            choice, ok = QInputDialog.getItem(self, '确认转换结果', label, choices, 0, False)
            if not ok:
                self.status.setText('未载入影像。转换结果已保留在：' + result['output']); return
            candidate = candidates[choices.index(choice)]
        else:
            candidate = candidates[0]
        self.selected_path = candidate['path']
        self.accept()

    def reject(self, *_args):
        if self.job is not None:
            self.closing = True; self.job.requestInterruption()
            self.close_button.setEnabled(False); self.status.setText('正在停止导入，请稍候…')
            return
        super().reject()

    def closeEvent(self, event):
        if self.job is not None:
            self.reject(); event.ignore()
        else:
            super().closeEvent(event)
