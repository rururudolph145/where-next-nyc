# Python Code Review Checklist

## 1. Correctness

- [ ] **Logic errors** — Does the code do what it claims? Check boundary conditions, off-by-one errors, and incorrect operator precedence.
- [ ] **Mutable default arguments** — Never use `def f(x=[])` or `def f(x={})`. Default mutable objects are shared across all calls.
- [ ] **Exception swallowing** — Bare `except:` or `except Exception: pass` hides bugs silently.
- [ ] **Off-by-one in slices/ranges** — Verify `range(n)`, `[:n]`, and `[n:]` boundaries match intent.
- [ ] **Floating-point comparison** — Never use `==` to compare floats. Use `math.isclose()` or a tolerance check.
- [ ] **Boolean trap** — Functions with multiple `bool` parameters (e.g. `process(True, False, True)`) are unreadable and error-prone. Use enums or keyword-only args.

## 2. Security

- [ ] **Injection risk** — No string formatting into SQL queries, shell commands, or file paths from user input. Use parameterised queries and `subprocess` with list args.
- [ ] **Hardcoded secrets** — No passwords, API keys, or tokens in source code. Use environment variables or a secrets manager.
- [ ] **Path traversal** — User-supplied file paths must be validated/sandboxed (e.g. `Path(base) / user_input` with `.resolve().is_relative_to(base)`).
- [ ] **Unsafe deserialization** — `pickle.loads` on untrusted data is a remote code execution risk. Prefer JSON or a safe format.
- [ ] **Insecure randomness** — Use `secrets` module for security-sensitive randomness, not `random`.

## 3. Type Safety

- [ ] **Missing type hints** — All function signatures (parameters and return type) should have annotations.
- [ ] **Implicit `None` return** — Functions that sometimes return a value and sometimes return nothing implicitly should declare `-> T | None` explicitly.
- [ ] **Unguarded `Optional` access** — Accessing attributes on a value that could be `None` without checking first causes `AttributeError`.
- [ ] **`Any` overuse** — Avoid `Any` as a type hint shortcut. Use `Union`, `TypeVar`, or a proper type.

## 4. Error Handling

- [ ] **Too broad exceptions** — Catch the most specific exception type available, not `Exception` or `BaseException`.
- [ ] **Missing cleanup** — Resources (files, DB connections, sockets) must be closed even on failure. Use `with` statements or `try/finally`.
- [ ] **Silent failures** — Functions that can fail should either raise or return a typed result. Never return `None` for both success and failure.
- [ ] **Error messages** — Exception messages should include context (what was attempted, what value caused the error).

## 5. Performance

- [ ] **Quadratic loops** — Avoid `O(n²)` patterns like nested loops over the same list, or membership checks (`x in list`) inside a loop. Use sets/dicts for O(1) lookup.
- [ ] **Repeated computation in loop** — Move invariant expressions outside loops (e.g. `len(arr)`, regex compilation, DB queries).
- [ ] **String concatenation in loop** — Use `"".join(parts)` instead of `result += s` inside a loop.
- [ ] **Loading entire file into memory** — Use generators or chunked reads for large files instead of `file.readlines()` or `pd.read_csv` without `chunksize`.
- [ ] **Unnecessary DataFrame copies** — Avoid chained pandas assignment (`df[...][...] = x`). Use `.loc` to avoid `SettingWithCopyWarning` and extra copies.

## 6. Readability & Style

- [ ] **Naming** — Variables, functions, and classes should have descriptive names. Avoid single letters except for indices (`i`, `j`) and coordinates (`x`, `y`).
- [ ] **Function length** — Functions longer than ~40 lines are a signal to decompose. Each function should do one thing.
- [ ] **Magic numbers** — Unexplained numeric literals (`if speed > 37.28`) should be named constants with a comment explaining their origin.
- [ ] **Dead code** — Remove commented-out code, unused imports, and unreachable branches.
- [ ] **Docstrings** — Public functions, classes, and modules should have docstrings explaining purpose, args, and return value (Google style).
- [ ] **Inconsistent style** — Follow PEP 8: snake_case for functions/variables, PascalCase for classes, UPPER_SNAKE for constants.

## 7. Structure & Design

- [ ] **God functions** — A single function doing I/O, business logic, and formatting violates single responsibility. Split concerns.
- [ ] **Tight coupling** — Functions that hardcode file paths, global state, or external service calls are hard to test. Pass dependencies as arguments.
- [ ] **Repeated code** — Three or more near-identical blocks are a signal to extract a helper function.
- [ ] **Global mutable state** — Module-level mutable variables that are modified by functions are a source of subtle bugs. Prefer passing state explicitly.
- [ ] **Premature abstraction** — Do not create base classes, registries, or plugin systems unless there are at least two concrete use cases now.

## 8. Testing Considerations

- [ ] **Untestable code** — Functions that mix I/O with logic cannot be unit tested. Separate pure computation from side effects.
- [ ] **No edge case handling** — Check whether empty input, zero, `None`, and maximum values are handled correctly.
- [ ] **Non-determinism** — Functions that depend on current time, random state, or external APIs should accept these as injectable parameters for testability.
