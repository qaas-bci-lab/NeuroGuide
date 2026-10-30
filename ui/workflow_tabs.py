"""A workflow rail with a fixed tab bar and action above scrollable inputs."""
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTabBar, QStackedWidget

class WorkflowTabs(QWidget):
    currentChanged = pyqtSignal(int)

    def __init__(self,parent=None):
        super().__init__(parent)
        self._layout=QVBoxLayout(self); self._layout.setContentsMargins(0,0,0,0); self._layout.setSpacing(7)
        self.bar=QTabBar(); self.bar.setExpanding(False); self.bar.setDrawBase(False)
        self.pages=QStackedWidget(); self._layout.addWidget(self.bar); self._layout.addWidget(self.pages,1)
        self.bar.currentChanged.connect(self._set_page)

    def addTab(self,widget,title):
        index=self.pages.addWidget(widget); self.bar.addTab(title); return index

    def set_actions(self,actions):
        self._layout.addWidget(actions)
        self.currentChanged.connect(actions.setCurrentIndex)

    def _set_page(self,index):
        self.pages.setCurrentIndex(index); self.currentChanged.emit(index)

    def setCurrentIndex(self,index): self.bar.setCurrentIndex(index)
    def currentIndex(self): return self.bar.currentIndex()
    def setTabEnabled(self,index,enabled): self.bar.setTabEnabled(index,enabled)
    def isTabEnabled(self,index): return self.bar.isTabEnabled(index)
    def widget(self,index): return self.pages.widget(index)
