# Testing Rules

## Testing Strategy

- Test framework: pytest
- Test file location: `tests/test_<module>.py`
- Coverage target: 80% or more for new code

## Test Structure

- AAA pattern: Arrange → Act → Assert
- Test name: `test_<function>_<condition>_<expected_result>`
- One test = one assertion (use parameterization when multiple assertions are needed)

## Required Test Cases

1. **Happy path**: Main input/output patterns
2. **Error cases**: Invalid input, exception cases
3. **Boundary values**: Empty lists, None, maximum values
4. **External APIs**: Mock with `unittest.mock`

## Forbidden Practices

- Do not connect to external APIs or databases from within tests
- Do not share state between tests (each test must be independent)
- Do not use `time.sleep` in tests (mock it instead)

## TDD Flow

1. RED: Write a failing test first
2. GREEN: Implement the minimum code needed to pass
3. REFACTOR: Refactor while keeping tests green
