# Python Ruff Conda Template

ETHZ Lab Python project template with ruff linting/formatting and pre-commit hooks.

## Quick Setup

### Prerequisites
- [Miniforge](https://github.com/conda-forge/miniforge) installed
- VS Code with Python extension

### One-Command Setup

**First, navigate to the project directory:**
```bash
cd python-ruff-conda-template
```

**Then run the setup script:**

**macOS/Linux:**
```bash
./setup.sh
```

**Windows:**
```bash
setup.bat
```

### Manual Setup (if scripts don't work)

**Make sure you're in the project directory first:**
```bash
cd python-ruff-conda-template
```

1. **Create environment:**
   ```bash
   conda env create -f environment.yml
   conda activate python-ruff-template
   ```

2. **Install pre-commit:**
   ```bash
   pre-commit install
   ```

3. **Configure VS Code:**
   - Open this project in VS Code - it will prompt you to install recommended extensions
   - Install: `charliermarsh.ruff` (Ruff) and `ms-python.mypy-type-checker` (MyPy) extensions in VS Code
   - Copy settings of `.vscode/settings_template.json` at bottom of existing `.vscode/settings.json` (macOS/Linux) or to `%APPDATA%\Code\User\settings.json` (Windows)

## Usage

**Make sure you're in the project directory and environment is activated:**
```bash
cd python-ruff-conda-template
conda activate python-ruff-template
```

### Code Quality
```bash
ruff check .          # Check for issues
ruff check --fix .    # Fix auto-fixable issues
ruff format .         # Format code
mypy .                # Type checking
```

### Testing
```bash
python src/example.py  # Run example
```

## What's Included

- **Python 3.11.8** via conda-forge
- **Ruff** for fast linting/formatting
- **Pre-commit hooks** for automated quality checks
- **VS Code integration** with consistent settings
- **Example code** (intentionally messy to demonstrate ruff)

## Project Structure

```
├── .vscode/
│   ├── extensions.json          # Recommended VS Code extensions
│   └── settings_template.json   # VS Code settings template
├── src/                         # Your source code
├── tests/                       # Your tests
├── environment.yml              # Conda environment
├── pyproject.toml               # Project config & ruff settings
├── setup.sh / setup.bat         # One-command setup
└── README.md                    # This file
```

## Adding Dependencies

Edit `environment.yml`:
```yaml
dependencies:
  - python=3.11.8
  - numpy  # Add conda packages here
  - pip
  - pip:
    - ruff>=0.1.0
    - pre-commit>=3.0.0
    - mypy>=1.17.1  # Add pip packages after here
```

Then run:
```bash
conda env update -f environment.yml
```

## Troubleshooting

- **Ruff not found in VS Code:** Restart VS Code after activating the conda environment
- **Pre-commit not working:** Run `pre-commit install` again
- **Environment issues:** Delete and recreate: `conda env remove -n python-ruff-template && conda env create -f environment.yml`

## Customizing the Template

### 1. Update Project Information

After cloning this template, update the project information in `pyproject.toml`:

```toml
[project]
name = "your-project-name"  # Change this to your project name
version = "0.1.0"
description = "Your project description here"
authors = [
    {name = "Your Name", email = "your.email@ethz.ch"}  # Update with your details
]
```

### 2. Create Your Private Repository

1. Create a new **private** repository on GitHub (e.g., `your-project-name`)
2. Clone this template: `git clone https://github.com/IDEALLab/python-ruff-conda-template.git your-project-name`
3. Remove the original git history: `rm -rf .git`
4. Initialize new git repository: `git init`
5. Add your new repository as remote: `git remote add origin https://github.com/YOUR_USERNAME/your-project-name.git`
6. Add all files: `git add .`
7. Make initial commit: `git commit -m "Initial commit from IDEAL Lab template"`
8. Push to your repository: `git push -u origin main`

### 3. Update Repository References

Update the following files to point to your new repository:

#### `pyproject.toml` (line 6):
```toml
name = "your-project-name"  # Change from "python-ruff-template"
```

#### `README.md` (line 2):
```markdown
# Your Project Name  # Change from "Python Ruff Conda Template"
```

#### `README.md` (line 4):
```markdown
Your project description here.  # Change from "ETHZ Lab Python project template..."
```

#### `README.md` (line 152):
```bash
git clone https://github.com/YOUR_USERNAME/your-project-name.git your-project-name
```

#### `README.md` (line 155):
```bash
git remote add origin https://github.com/YOUR_USERNAME/your-project-name.git
```

#### `environment.yml` (line 1):
```yaml
name: your-project-name  # Change from "python-ruff-template"
```

#### `setup.sh` and `setup.bat` (line 40):
Update the environment name in the activation command:
```bash
conda activate your-project-name  # Change from "python-ruff-template"
```

## Support

For issues, check the troubleshooting section above. This template is designed to be self-contained and not require external support.
