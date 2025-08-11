import os

from cx_Freeze import Executable, setup

executables = [
    Executable(
        "build_exe.py",
        target_name="mountain3d.exe",
        icon=os.path.join("3d-python-media", "other", "icon.ico"),
        shortcut_name="mountain3d",
        shortcut_dir="DesktopFolder",
    )
]


include_files = [
    ("3d-python-media", os.path.join("lib", "3d-python-media")),
    ("config_annotation", os.path.join("lib", "config_annotation")),
]

options = {
    "build_exe": {
        "include_msvcr": True,
        "include_files": include_files,
        "build_exe": "build_executable",
    },
}

setup(
    name="mountain3d",
    version="0.0.1",
    description="mountain3d program",
    executables=executables,
    options=options,
)
