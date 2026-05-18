---
name: refactor
description: Refactors specified code. Removes dead code, splits functions, and improves naming.
allowed-tools: Read, Glob, Grep, Edit, Write, Bash
---

# Refactoring Assistant

Refactor the specified code.

## Principles

- **Preserve behavior**: Refactoring is behavior-preserving changes only
- **Small, incremental steps**: Make small changes at a time, not big rewrites
- **Follow existing patterns**: Respect the conventions used in the project

## Checklist

1. **Remove dead code**: Delete unused imports, variables, and functions
2. **Split functions**: Break overly long functions (50+ lines) by responsibility
3. **Improve naming**: Rename unclear variables and functions
4. **Eliminate duplication**: Consolidate copy-pasted code
5. **Type safety**: Add clear type annotations (only when existing code already has types)

## Procedure

1. Read the target code and understand the current structure
2. Present a refactoring plan
3. Apply the changes after user approval
4. Run ruff check to confirm no syntax errors

Target: $ARGUMENTS
