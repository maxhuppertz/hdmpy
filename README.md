# hdmpy

The hdmpy package is a Python port of parts of the R package [hdm](https://github.com/cran/hdm). All credit for the original package goes to the authors of hdm, all mistakes are my own. This project is in its very early stages and documentation is virtually nonexistent, so use it at your own risk.

## Installation

### Using pip (recommended)

```bash
pip install hdmpy
```

### Using uv

```bash
uv add hdmpy
```

### From source (pip)

```bash
git clone https://github.com/maxhuppertz/hdmpy.git
cd hdmpy
pip install .
```

### Editable install for development (pip)

```bash
pip install -e .[dev]
```

After installation, you can import the package:

```python
import hdmpy as hdm
```
