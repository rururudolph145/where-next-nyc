---
name: test
description: Generates and runs tests. Creates unit tests for existing code and runs them with pytest.
allowed-tools: Read, Glob, Grep, Bash, Edit, Write
---

# Test Generator & Runner

Generate or run tests.

## Mode Selection

- Argument `run` → Run existing tests
- Argument is a file path → Generate tests for that file
- No argument → Run tests for changed files

## Test Generation Rules

1. **Test file location**: Create in the `tests/` directory as `test_<module>.py`
2. **Framework**: Use pytest
3. **Coverage targets**:
   - Happy path: Main input/output patterns
   - Error cases: Invalid input, exception cases
   - Boundary values: Empty, maximum, None
4. **Mocking**: Mock external API calls with `unittest.mock`
5. **Naming**: Use the format `test_<function>_<condition>`

## Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Specific file
python -m pytest tests/test_xxx.py -v

# With coverage
python -m pytest tests/ --cov=. --cov-report=term-missing
```

Target: $ARGUMENTS
