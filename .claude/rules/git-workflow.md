# Git Workflow Rules

## Branch Strategy

- `main` - Production environment. Do not push directly.
- `dev_MVP` - Integration branch. All feature branches merge here first.
- `feature/*` or `bug_fix_*` - Feature development and bug fix branches
- Merge via GitLab/GitHub merge requests

## Merge Targets

- **Never merge a `feature/*` or `bug_fix_*` branch directly to `main`.**
- Feature branches MUST target `dev_MVP` as the PR base.
- `dev_MVP` → `main` is a separate release PR, performed only when a set of integrated features is ready for production.
- When opening a PR (e.g., via `gh pr create`), always pass `--base dev_MVP` for feature work.
- If a PR is found targeting `main` from a `feature/*` branch, change its base to `dev_MVP` before merging (`gh pr edit <num> --base dev_MVP`).

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
