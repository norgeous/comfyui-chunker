# Tests

Test files follow the naming scheme `<module>_test.py`, where `<module>` corresponds to the module being tested.

## Running Tests

### Run all tests
```bash
python -m pytest lib/tests/ -v --rootdir=lib
```

### Run specific test file
```bash
python -m pytest lib/tests/test_utils_blend_mode.py -v --rootdir=lib
python -m pytest lib/tests/test_mux.py -v --rootdir=lib
```

### Run tests with coverage (optional)
```bash
python -m pytest lib/tests/ -v --rootdir=lib --cov=lib
```
