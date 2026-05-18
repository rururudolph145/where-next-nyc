---
name: deploy-check
description: Pre-deployment checklist. Verifies code quality, configuration, and security.
allowed-tools: Read, Glob, Grep, Bash
---

# Pre-Deployment Checklist

Perform pre-deployment verification.

## Check Items

### 1. Code Quality
```bash
# Python lint
ruff check .
# TypeScript type check (if there is a frontend)
npx tsc --noEmit
# Frontend build (if there is a frontend)
npm run build
```

### 2. Git Status
- [ ] No unintended changes are staged (`git status`)
- [ ] No .env, credentials, or secrets are included
- [ ] No large binary files are included
- [ ] No debug print / console.log statements remain

### 3. Configuration Files
- [ ] Environment variables are properly set
- [ ] API endpoint configuration is correct
- [ ] Resource limits (K8s manifests, etc.) are appropriate

### 4. Security
- [ ] No hardcoded API keys or tokens
- [ ] CORS configuration is appropriate
- [ ] No authentication / authorization bypasses
- [ ] `verify=False` is not used

### 5. Tests
- [ ] All tests pass
- [ ] Coverage meets the required threshold

## Output

Report Pass/Fail for each item, and suggest fixes for any failures.
