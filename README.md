# Splitter Python code

## 1. Testing
### 1.1 Using
- Install *pytest* and *pytest-cov* for testing and coverage-tesing.<br>
`pip install pytest pytest-cov`
- All tests are in path: *tests*<br>
- To **start testing** use: <br>
`python -m pytest tests/`<br>
- To **check coverage** use: <br>
`python -m pytest --cov=source tests/`
- To create **html report** use (check *htmlcov/index.html* after): <br>
`python -m pytest --cov=source --cov-report=html tests/`

### 1.2 Test requirements:
- Correctness `tests/test_correctness.py`
    - Changes in source code **don't affect** the result of splitting
- Exceptions handling `tests/test_exceptions.py`
    - Exceptions are thrown on *incorrect input* to splitter
    - Exceptions are thrown on problems during splitting with *stratification*
    - ? Exceptions have **user-friendly** message for front-end
