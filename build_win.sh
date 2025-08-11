#!/bin/bash
set -ex

sudo apt install wine

wine winecfg -v win10

USERNAME=$(whoami)
PYTHON_PATH=~/.wine/drive_c/users/$USERNAME/Local\ Settings/Application\ Data/Programs/Python/Python39

if [ ! -d "$PYTHON_PATH" ]; then
    # Download & Install Python 3.9
    if [ ! -f "./python-3.9.0-amd64.exe" ]; then
        wget https://www.python.org/ftp/python/3.9.0/python-3.9.0-amd64.exe
    fi

    # Important: Click 'Add Python 3.9 to PATH' and select 'Install Now'
    wine cmd /c python-3.9.0-amd64.exe
fi

wine python.exe "$PYTHON_PATH"/Scripts/pip.exe install poetry

POETRY_CACHE_DIR=~/.wine/drive_c/users/$USERNAME/Local\ Settings/Application\ Data/pypoetry/Cache

wine python.exe "$PYTHON_PATH"/Scripts/poetry.exe install

# Select environment which was created by poetry in Windows (!!)
ENV_NAME=$(ls "$POETRY_CACHE_DIR/virtualenvs" | grep mountain3d | tr -d '\r')
rm -rf build_executable && wine "$POETRY_CACHE_DIR"/virtualenvs/"$ENV_NAME"/Scripts/python.exe setup_freeze.py build
