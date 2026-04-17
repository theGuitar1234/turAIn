### Installment

**Configuration
- src/main/resources/config
    -requirements-dev.txt
    -requirements.txt
```
python -m pip install --no-user -r requirements<profile>.txt
```

**pyproject.toml
- Install CUDA
```
pip install ".[cuda12]"
pip install ".[dev]"
pip install ".[dev,cuda12]"
```