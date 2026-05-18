# Coding Style Rules

## Core Principles

- **KISS**: Choose the simplest possible implementation
- **YAGNI**: Don't build what isn't needed now
- **DRY**: Only consider deduplication after 3+ repetitions (2 is acceptable)

## Python

- Follow ruff (line-length=88)
- Keep functions under 50 lines as a guideline
- Limit nesting to 3 levels; flatten deeper logic with early returns
- Comments should explain "why"; let the code express "what"
- Docstrings are required only for public APIs

## TypeScript

- Follow ESLint configuration
- `any` is prohibited; use `unknown` when the type is unknown
- Define component props using interfaces

## Naming Conventions

| Target | Python | TypeScript |
|--------|--------|-----------|
| Variables / functions | snake_case | camelCase |
| Classes | PascalCase | PascalCase |
| Constants | UPPER_SNAKE | UPPER_SNAKE |
| Files | snake_case.py | PascalCase.tsx / camelCase.ts |

## Git

- Commit messages may be in English or Japanese
- Emoji prefixes are recommended: `✨ feature`, `🐛 bugfix`, `♻️ refactor`, `📝 docs`, `💚 CI/CD`
