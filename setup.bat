@echo off
REM Setup script for Python Ruff Conda Template (Windows)
REM Run this script to set up the development environment

echo 🚀 Setting up Python Ruff Conda Template...

REM Check if conda is available
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ Conda not found. Please install miniforge first:
    echo    https://github.com/conda-forge/miniforge
    pause
    exit /b 1
)

echo ✅ Conda found

REM Create conda environment
echo 📦 Creating conda environment...
conda env create -f environment.yml

REM Activate environment
echo 🔄 Activating environment...
call conda activate python-ruff-template

REM Install pre-commit hooks
echo 🪝 Installing pre-commit hooks...
pre-commit install

REM Test the setup
echo 🧪 Testing the setup...
python src/example.py

echo.
echo 🎉 Setup complete!
echo.
echo Next steps:
echo 1. Activate the environment: conda activate python-ruff-template
echo 2. Configure VS Code:
echo    - Open project in VS Code (will prompt for extensions)
echo    - Install Ruff and MyPy extensions: charliermarsh.ruff and ms-python.mypy-type-checker
echo    - Copy settings at bottom of existing %%APPDATA%%\Code\User\settings.json
echo 3. Start coding! 🚀
echo.
echo Useful commands:
echo   ruff check .          # Check for issues
echo   ruff check --fix .    # Fix auto-fixable issues
echo   ruff format .         # Format code
echo   mypy .                # Type checking
echo   pre-commit run --all-files  # Run all pre-commit hooks

pause
