"""Shared desktop palette and control hierarchy for the imaging workspace."""
WINDOW_STYLE = '''
QMainWindow, QDialog { background: #15171c; }
QWidget { color: #e1e5ed; font-family: "Microsoft YaHei UI", "Microsoft YaHei", "Segoe UI"; font-size: 12px; }
QFrame#ctrlPanel { background: #1c1f26; border: 1px solid #353943; border-radius: 12px; }
QLabel { color: #a8afbd; }
QLabel#workspaceTitle { color: #f4f6fa; font-size: 19px; font-weight: bold; }
QPushButton { background: #282c35; color: #e1e5ed; border: 1px solid #414754; border-radius: 6px; padding: 7px 12px; }
QPushButton:hover { background: #353b46; border-color: #6f9ff0; }
QPushButton:focus, QToolButton:focus, QComboBox:focus, QLineEdit:focus { border: 1px solid #8ab4ff; }
QPushButton:disabled { background: #20232b; color: #69717f; border-color: #343943; }
QPushButton#primaryBtn, QPushButton#primary { background: #326bd1; color: white; border: 1px solid #497fe0; font-weight: bold; }
QPushButton#primaryBtn:hover, QPushButton#primary:hover { background: #447cdf; }
QPushButton#primaryBtn:disabled, QPushButton#primary:disabled { background: #20232b; color: #69717f; border-color: #343943; }
QToolButton#disclosure { color: #b0bdce; background: transparent; border: none; padding: 5px 0; }
QScrollArea#sidebarScroll { background: transparent; border: none; }
QSplitter::handle { background: #353943; width: 5px; }
QSplitter::handle:hover { background: #6f9ff0; }
QScrollBar:vertical { background: #1c1f26; width: 8px; margin: 0; }
QScrollBar::handle:vertical { background: #505866; border-radius: 4px; min-height: 24px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
QListWidget, QTreeWidget { background: #171a21; border: 1px solid #353943; border-radius: 7px; outline: none; }
QPlainTextEdit { background: #171b22; color: #bac6d8; border: 1px solid #3b424f; border-radius: 6px; }
QListWidget::item { padding: 4px 7px; border-radius: 3px; }
QListWidget::item:selected { background: #2a4165; }
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox { background: #171a21; border: 1px solid #414754; border-radius: 5px; padding: 5px 7px; }
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView { background: #1c1f26; color: #e1e5ed; selection-background-color: #2a4165; }
QSlider::groove:horizontal { background: #3e444f; height: 4px; border-radius: 2px; }
QSlider::handle:horizontal { background: #78a5f0; width: 13px; height: 13px; margin: -5px 0; border-radius: 6px; }
QCheckBox { spacing: 6px; }
QCheckBox::indicator { width: 15px; height: 15px; border-radius: 3px; border: 1px solid #626c7c; background: #1a1e27; }
QCheckBox::indicator:checked { background: #598ce8; }
QFrame#fieldInfo { background: #1c1f26; border: 1px solid #353943; border-radius: 8px; }
QProgressBar { background: #303641; border: none; border-radius: 3px; height: 6px; }
QProgressBar::chunk { background: #598ce8; border-radius: 3px; }
QStatusBar { background: #1c1f26; color: #c2cad6; }
'''

PANEL_STYLE = '''
QWidget#individualPanel { background: transparent; }
QLabel { color: #cbd1dd; font-size: 12px; }
QLabel#heading { color: #eef1f7; font-size: 17px; font-weight: bold; }
QLabel#muted { color: #a2abba; font-size: 11px; }
QFrame#card { background: #232730; border: 1px solid #3b424f; border-radius: 8px; }
QScrollArea, QScrollArea > QWidget > QWidget { background: #1c1f26; border: none; }
QTabWidget::pane { border: none; padding-top: 7px; }
QTabBar::tab { background: #252932; color: #b1bbc9; padding: 8px 11px; margin-right: 4px; border-radius: 5px; }
QTabBar::tab:selected { background: #2b4770; color: white; }
QTabBar::tab:disabled { color: #737d8e; }
QPlainTextEdit { background: #171b22; color: #bac6d8; border: 1px solid #3b424f; border-radius: 6px; }
'''
