"""The selected NeuroGuide identity, shared by the home and workspace."""
from pathlib import Path
import sys

from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QPainterPath, QPen, QColor
from PyQt6.QtWidgets import QLabel


WORDMARK = '<span style="color:#e1e5ed">Neuro</span><span style="color:#326bd1">Guide</span>'


def logo_path():
    root = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parents[1]))
    return root/'assets/branding/neuroguide_mark.png'


def application_icon():
    return QIcon(str(logo_path()))


class LogoLabel(QLabel):
    def paintEvent(self, event):
        if self.width() < 64 or self.pixmap() is None:
            super().paintEvent(event)
            return
        # Present the home mark as a rounded app tile; the compact workspace
        # mark uses ordinary QLabel rendering without an extra frame.
        painter = QPainter(self)
        painter.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect()).adjusted(.5, .5, -.5, -.5), 24, 24)
        painter.setClipPath(path)
        painter.drawPixmap(self.rect(), self.pixmap())
        painter.setClipping(False)
        painter.setPen(QPen(QColor('#353943'), 1))
        painter.drawPath(path)


def logo_label(size, parent=None, name='brandMark'):
    label = LogoLabel(parent)
    label.setObjectName(name)
    label.setFixedSize(size, size)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setToolTip('NeuroGuide · 个体电场建模与 TI 方案规划')
    label.setAccessibleName('NeuroGuide 标志')
    pixmap = QPixmap(str(logo_path()))
    ratio = label.devicePixelRatioF()
    pixmap = pixmap.scaled(round(size*ratio), round(size*ratio),
                           Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
    pixmap.setDevicePixelRatio(ratio)
    label.setPixmap(pixmap)
    return label
