---
name: coding-standards
description: Coding standards and best practices for Python/TypeScript. Apply during code writing, review, and refactoring.
allowed-tools: Read, Grep, Glob
---

# Coding Standards

## Python

### Style
- Formatter: ruff (line-length=88)
- Import order: stdlib > third-party > local (isort-compatible)
- Strings: prefer double quotes
- f-strings: use for variable interpolation

### Type Annotations
- Add type annotations to function signatures
- Use `X | None` (preferred over `Optional[X]`, Python 3.10+)
- Use `dict`, `list`, `tuple` (preferred over the capitalized versions, Python 3.9+)
- Name complex types with `TypeAlias`

### Error Handling
- No bare `except:` → specify concrete exceptions
- Include logging with `except Exception as e:`
- Use `tenacity` for retries

### Logging
- Use the `logger.bind(user_name=xxx)` pattern with loguru
- Levels: debug (development), info (normal), warning (recoverable), error (action required)
- Structured logs: `message=key_event...param1=value1`

### FastAPI
- Define endpoints with `async def`
- Use Pydantic models for requests/responses
- Specify constraints via `Field` for validation

## TypeScript / React

### Style
- Semicolons: required
- Indentation: 2 spaces
- Components: functional components + hooks

### Types
- `any` is prohibited → use `unknown` and narrow with type guards
- Define props with interface
- Define types and validation for API responses

### State Management
- Local state: `useState`
- Derived state: `useMemo`
- Side effects: `useEffect` (with accurate dependency arrays)

## Bad vs Good

```python
# Bad
def process(data):
    try:
        result = do_something(data)
    except:
        pass

# Good
def process(data: dict) -> Result | None:
    try:
        result = do_something(data)
    except ValueError as e:
        logger.error(f"Invalid data: {e}")
        return None
```
