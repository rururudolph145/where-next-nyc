---
name: code-review
description: Performs code review on specified files or git diff. Checks quality, security, and performance.
allowed-tools: Read, Glob, Grep, Bash
---

# Code Reviewer

Perform a code review.

## Identifying the Review Target

1. If a file path is given in the arguments, review that file
2. If no argument is given, retrieve the diff in this order: `git diff --cached` (staged) → `git diff` (unstaged)
3. If there is no diff either, review the changes in the latest commit

## Check Items

### Correctness
- [ ] No logic errors
- [ ] Edge cases are handled
- [ ] Error handling is appropriate

### Security
- [ ] No hardcoded secrets
- [ ] No injection vulnerabilities (SQL, XSS, command)
- [ ] Input validation is appropriate

### Performance
- [ ] No N+1 queries
- [ ] No unnecessary loops or recomputations
- [ ] No memory leak risks

### Maintainability
- [ ] Naming is clear
- [ ] Functions have a single responsibility
- [ ] No magic numbers

## Output Format

Report each issue in the following format:

```
[severity: critical/warning/info] file:line
Issue: description
Suggestion: how to fix
```

Output a summary at the end.

Target: $ARGUMENTS
