# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata


spec_root = Path(SPECPATH).resolve()
if (spec_root / "desktop_app.py").exists():
    project_root = spec_root
elif (spec_root.parent / "desktop_app.py").exists():
    project_root = spec_root.parent
else:
    project_root = Path.cwd().resolve()

datas = [
    (str(project_root / "app.py"), "."),
    (str(project_root / "README.md"), "."),
    (str(project_root / "使用说明.txt"), "."),
    (str(project_root / "requirements.txt"), "."),
]
binaries = []
hiddenimports = [
    "streamlit.web.cli",
    "streamlit.runtime.scriptrunner.magic_funcs",
    "streamlit.runtime.runtime",
    "webview",
    "webview.platforms.winforms",
    "webview.platforms.edgechromium",
    "fitz",
]

for module_name in [
    "streamlit.runtime.scriptrunner",
    "streamlit.runtime.scriptrunner_utils",
]:
    try:
        hiddenimports += collect_submodules(module_name)
    except Exception:
        pass

for package_name in ["streamlit", "pydeck", "altair"]:
    try:
        datas += collect_data_files(package_name)
        datas += copy_metadata(package_name)
    except Exception:
        pass

for distribution_name in [
    "pandas",
    "numpy",
    "pillow",
    "requests",
    "PyMuPDF",
    "pywebview",
    "protobuf",
    "pyarrow",
]:
    try:
        datas += copy_metadata(distribution_name)
    except Exception:
        pass


a = Analysis(
    [str(project_root / "desktop_app.py")],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib.tests",
        "numpy.tests",
        "pandas.tests",
        "PIL.tests",
        "pytest",
        "IPython",
        "notebook",
        "jupyter",
        "jupyterlab",
        "sphinx",
        "docutils",
        "torch",
        "torchvision",
        "torchaudio",
        "tensorflow",
        "scipy",
        "sklearn",
        "skimage",
        "cv2",
        "numba",
        "llvmlite",
        "dask",
        "distributed",
        "xarray",
        "astropy",
        "gradio",
        "panel",
        "bokeh",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AutoPDFTranslator",
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
    name="AutoPDFTranslator",
)
