---
name: tdd
description: Test-driven development workflow. Implement using the RED → GREEN → REFACTOR cycle.
allowed-tools: Read, Glob, Grep, Bash, Edit, Write
---

# TDD Workflow

Implement $ARGUMENTS using test-driven development.

## Procedure

### 1. Define the Interface
- Define the signature of the target function/class
- Clarify input and output types

### 2. RED (Write a failing test)
- Create tests in `tests/test_<module>.py`
- Write test cases for happy path, error cases, and boundary values
- Run `python -m pytest tests/test_<module>.py -v` and confirm failure

### 3. GREEN (Minimal implementation to pass tests)
- Implement the minimum amount of code needed to pass the tests
- Run tests again to confirm everything passes

### 4. REFACTOR (Refactoring)
- Improve the code while keeping the tests green
- Improve naming, structure, and remove duplication
- Run `ruff check` for quality verification

### 5. Coverage Check
```bash
python -m pytest tests/test_<module>.py --cov=. --cov-report=term-missing -v
```

Report progress at each step.

Target: $ARGUMENTS
