"""Sidebar disclosures that overlay subsequent controls without moving them."""
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea


class SidebarDrawer(QFrame):
    def __init__(self, section):
        super().__init__(section.window())
        self.section=section
        self.setObjectName('sidebarDrawer')
        from ui.theme import PANEL_STYLE
        self.setStyleSheet(PANEL_STYLE+'QFrame#sidebarDrawer {background:#20242d;border:1px solid #5276ae;border-radius:8px;}')
        layout=QVBoxLayout(self);layout.setContentsMargins(10,10,10,10)
        row=QHBoxLayout();row.addWidget(QLabel(section.toggle.text()),1)
        close=QPushButton('收起');close.clicked.connect(lambda *_args:section.set_expanded(False));row.addWidget(close)
        layout.addLayout(row)
        self.scroll=QScrollArea();self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setWidget(section.body);layout.addWidget(self.scroll)
        section.window().installEventFilter(self)
        sidebar=getattr(section.window(),'sidebar_scroll',None)
        if sidebar:sidebar.viewport().installEventFilter(self)

    def place(self):
        owner=self.section.window()
        sidebar=getattr(owner,'sidebar_scroll',None)
        anchor=sidebar.viewport() if sidebar else self.section
        corner=anchor.mapTo(owner,anchor.rect().topLeft())
        top=self.section.toggle.mapTo(owner,self.section.toggle.rect().bottomLeft()).y()+5
        bottom=min(owner.height()-10,corner.y()+anchor.height()-8)
        height=min(max(140,self.section.body.sizeHint().height()+66),max(140,bottom-top))
        top=max(corner.y()+8,min(top,bottom-height))
        self.setGeometry(corner.x()+6,top,max(220,anchor.width()-12),height)

    def eventFilter(self,obj,event):
        if self.isVisible() and event.type()==QEvent.Type.Resize:self.place()
        if self.isVisible() and event.type()==QEvent.Type.KeyPress and event.key()==Qt.Key.Key_Escape:
            self.section.set_expanded(False);return True
        return super().eventFilter(obj,event)

    def reveal(self):
        owner=self.section.window()
        previous=getattr(owner,'_open_sidebar_drawer',None)
        if previous and previous is not self:previous.section.set_expanded(False)
        owner._open_sidebar_drawer=self
        self.section.body.show();self.place();self.show();self.raise_()
