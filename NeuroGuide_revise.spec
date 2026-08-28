# -*- mode: python ; coding: utf-8 -*-

_block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('sample_data', 'sample_data'),
        ('mri', 'mri'),
        ('assets', 'assets'),
    ],
    hiddenimports=[
        'numpy',
        'scipy',
        'scipy.ndimage',
        'scipy.optimize',
        'nibabel',
        'nibabel.processing',
        'SimpleITK',
        'matplotlib',
        'matplotlib.backends.backend_qtagg',
        'PyQt6',
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
        'pyvista',
        'pyvistaqt',
        'vtk',
        'vtkmodules',
        'vtkmodules.all',
        'mpl_toolkits',
        'mpl_toolkits.axes_grid1',
        'mpl_toolkits.axes_grid1.inset_locator',
        'tools.register_template_to_individual',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PyQt5'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='NeuroGuide ver3.5',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='NeuroGuide ver3.5',
)
