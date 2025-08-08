# hdmpy

The hdmpy package is a Python port of parts of the R package [hdm](https://github.com/cran/hdm). All credit for the original package goes to the authors of hdm, all mistakes are my own. This project is in its very early stages and documentation is virtually nonexistent, so use it at your own risk.

## Installation

### Using uv (recommended)

- Create and use a virtual environment, and install dependencies:

```bash
uv venv
source .venv/bin/activate  # or `uv venv --python 3.11 && source .venv/bin/activate`
uv pip install -e .[dev]
```

- Build a wheel and install locally [optional]:

```bash
uv build
uv pip install dist/*.whl
```

### From source (pip)

1) Clone the repository

2) Build and install using a modern build backend:

```bash
python -m pip install --upgrade pip build
python -m build
python -m pip install dist/*.whl
```

### Editable install for development (pip)

```bash
python -m pip install -e .[dev]
```

After installation, you can import the package:

```python
import hdmpy as hdm
```
