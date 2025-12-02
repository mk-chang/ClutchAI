# Conda Environment Setup for Pytest in Cursor

This guide explains how to set up a conda environment for running pytest tests in Cursor IDE.

## Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) installed
- Cursor IDE installed

## Step 1: Create the Conda Environment

From the project root directory, create the conda environment using the provided `environment.yml`:

```bash
conda env create -f environment.yml
```

This will create an environment named `ClutchAI` with all the required dependencies.

## Step 2: Activate the Environment

Activate the conda environment:

```bash
conda activate ClutchAI
```

**Note for macOS/Linux**: You may need to initialize conda for your shell first:
```bash
conda init bash  # or zsh, fish, etc.
```

Then restart your terminal or run `source ~/.bashrc` (or `~/.zshrc` for zsh).

## Step 3: Configure Cursor to Use the Conda Environment

### Option A: Using Cursor's Settings (Recommended)

1. Open Cursor Settings:
   - **macOS**: `Cmd + ,` or `Cursor > Settings`
   - **Windows/Linux**: `Ctrl + ,` or `File > Preferences > Settings`

2. Search for "Python Interpreter" or "Python Path"

3. Set the Python interpreter to your conda environment:
   - Click "Select Interpreter" or browse to the interpreter path
   - The conda environment path is typically:
     - **macOS/Linux**: `~/anaconda3/envs/ClutchAI/bin/python` (or `~/miniconda3/envs/ClutchAI/bin/python`)
     - **Windows**: `C:\Users\<username>\anaconda3\envs\ClutchAI\python.exe`

   Or use the path where conda installed it (find it with `conda env list`):
   ```bash
   conda env list
   ```

### Option B: Using Workspace Settings

1. In Cursor, press `Cmd/Ctrl + Shift + P` to open the command palette
2. Type "Python: Select Interpreter"
3. Choose the `ClutchAI` conda environment from the list
4. If it doesn't appear, click "Enter interpreter path" and browse to:
   ```
   ~/anaconda3/envs/ClutchAI/bin/python
   ```
   (Adjust path based on your conda installation)

### Option C: Create a `.vscode/settings.json` file

Create a `.vscode/settings.json` file in the project root (Cursor uses VS Code settings):

```json
{
  "python.defaultInterpreterPath": "${env:HOME}/miniconda3/envs/ClutchAI/bin/python",
  "python.testing.pytestEnabled": true,
  "python.testing.unittestEnabled": false,
  "python.testing.pytestArgs": [
    "tests"
  ]
}
```

**Adjust the path** based on your conda installation location (replace `miniconda3` with `anaconda3` if needed, or use the actual path from `conda env list`).

## Step 4: Verify the Setup

1. **Check Python interpreter in Cursor**:
   - Look at the bottom-right corner of Cursor - it should show the Python version and environment name
   - Click on it to change if needed

2. **Verify pytest installation**:
   ```bash
   conda activate ClutchAI
   pytest --version
   ```

3. **Run a test to verify everything works**:
   ```bash
   pytest tests/ -v
   ```

## Step 5: Configure Environment Variables (Optional for Tests)

Some tests may require environment variables. Create a `.env` file from the example:

```bash
cp env.example .env
```

Edit `.env` and add your test credentials (tests use mocks, so you may not need real keys).

## Running Tests in Cursor

### Method 1: Using the Test Runner UI

1. Open any test file (e.g., `tests/test_fantasy_news.py`)
2. You should see "Run Test" or "Debug Test" buttons above test functions
3. Click to run individual tests or the entire test suite

### Method 2: Using the Terminal

Open the integrated terminal in Cursor (`Ctrl + \` or `View > Terminal`) and run:

```bash
# Activate the environment first
conda activate ClutchAI

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/test_fantasy_news.py

# Run a specific test
pytest tests/test_fantasy_news.py::TestFantasyNewsTool::test_init
```

### Method 3: Using Cursor's Command Palette

1. Press `Cmd/Ctrl + Shift + P`
2. Type "Test: Run All Tests" or "Test: Run Tests in Current File"
3. Select the command

## Troubleshooting

### Conda environment not showing in Cursor

1. **Check conda is initialized**:
   ```bash
   conda init
   ```

2. **Verify environment exists**:
   ```bash
   conda env list
   ```

3. **Restart Cursor** after creating/activating the environment

### Pytest not found

1. **Reinstall pytest in the conda environment**:
   ```bash
   conda activate ClutchAI
   pip install pytest pytest-cov pytest-mock
   ```

### Import errors in tests

1. **Ensure the project root is in Python path** (already configured in `pytest.ini`):
   ```ini
   pythonpath = .
   ```

2. **Verify imports work**:
   ```bash
   conda activate ClutchAI
   python -c "from ClutchAI.tools.fantasy_news import FantasyNewsTool; print('OK')"
   ```

### Python interpreter path issues

Find your exact conda environment path:
```bash
conda activate ClutchAI
which python
# or
python -c "import sys; print(sys.executable)"
```

Then use that exact path in Cursor's Python interpreter settings.

## Updating the Environment

If you add new dependencies:

1. **Update `requirements.txt`**:
   ```bash
   pip freeze > requirements.txt
   ```

2. **Update `environment.yml`** manually or regenerate:
   ```bash
   conda env export > environment.yml
   ```

3. **Update the conda environment**:
   ```bash
   conda env update -f environment.yml --prune
   ```

## Quick Reference

```bash
# Create environment
conda env create -f environment.yml

# Activate environment
conda activate ClutchAI

# Run all tests
pytest

# Run specific test file
pytest tests/test_fantasy_news.py -v

# Run with coverage
pytest --cov=ClutchAI --cov-report=html

# Deactivate environment
conda deactivate

# Remove environment (if needed)
conda env remove -n ClutchAI
```

