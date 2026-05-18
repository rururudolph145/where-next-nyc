# Git Workflow Rules

## Branch Strategy

- `main` - Production environment. Do not push directly.
- `feature/*` or `bug_fix_*` - Feature development and bug fix branches
- Merge via GitLab/GitHub merge requests

## Commits

- One commit = one logical change
- Before committing, run `git diff --cached` to confirm there are no unintended changes
- Never commit `.env`, secrets, or large binary files

## Forbidden Operations

- `git push --force` to main/master
- `git reset --hard` (risk of data loss)
- `git rebase` on shared branches

## Merge Conflicts

- Resolve conflicts manually, understanding the intent of both changes before merging
- Do not rely on automatic merges; verify the result
