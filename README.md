### Installment

*Create virtual Environment
```
python -m venv .venv
```
- Activate Environment
```
.venv\Scripts\activate
```
- Get Help
```
help
```

*Configuration
- src/main/resources/config
    -requirements-dev.txt
    -requirements.txt

```
pip install --upgrade build
python -m pip install --no-user -r requirements<profile>.txt
```
*pyproject.toml
-- Install CUDA (cuda12 or cuda13)
```
pip install ".[cuda12]"
pip install ".[dev]"
pip install ".[dev,cuda12]"
```

*Project Build
```
python -m build
```
