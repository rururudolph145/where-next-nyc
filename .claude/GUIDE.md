# .claude Configuration Guide

## Directory Overview

```
.claude/
├── claude.md                          # Instructions for Claude (read first)
├── settings.json                      # Model, permissions, and hooks configuration
├── GUIDE.md                           # This file
│
├── agents/                            # Subagent definitions
│   ├── planner.md                     #   Implementation planning
│   ├── code-reviewer.md               #   Code quality review
│   └── security-reviewer.md           #   Security vulnerability review
│
├── commands/                          # Slash commands (invoked with /xxx)
│   ├── plan.md                        #   /plan - implementation planning
│   ├── code-review.md                 #   /code-review - code review
│   ├── build-fix.md                   #   /build-fix - fix build errors
│   ├── log-analyze.md                 #   /log-analyze - log analysis
│   ├── refactor.md                    #   /refactor - refactoring
│   ├── test.md                        #   /test - test generation and execution
│   ├── tdd.md                         #   /tdd - test-driven development
│   └── deploy-check.md                #   /deploy-check - pre-deployment check
│
├── rules/                             # Always-applied rules (auto-loaded)
│   ├── coding-style.md                #   Coding style (KISS/YAGNI/DRY)
│   ├── security.md                    #   Security rules
│   ├── testing.md                     #   Testing strategy (pytest, AAA, TDD)
│   ├── git-workflow.md                #   Git workflow rules
│   ├── performance.md                 #   Performance rules
│   └── agents.md                      #   Subagent usage rules
│
└── skills/                            # Auto-triggered skills (applied when conditions match)
    ├── api-patterns/SKILL.md          #   FastAPI design patterns
    ├── coding-standards/SKILL.md      #   Coding conventions
    ├── frontend-patterns/SKILL.md     #   React/TypeScript patterns
    └── security-review/SKILL.md       #   OWASP security checks
```

---

## Role of Each Component

### 1. `claude.md` - Instructions
**When it is read**: Always loaded at the start of a conversation
**Role**: Tells Claude "what to do and what not to do in this project"

Current contents:
- Prohibitions (no killing processes, no hardcoded secrets)
- Commonly used commands (Python/TypeScript/tests)
- List of available slash commands

**Customization**: For new projects, append the tech stack and directory structure at the end of the file

### 2. `settings.json` - Execution Environment
**When it is read**: Auto-applied when Claude Code starts

| Setting | Description |
|---------|-------------|
| `model` | `claude-opus-4-7` (use Opus for all sessions) |
| `permissions.allow` | Commands that run without confirmation (uv, ruff, git status, etc.) |
| `permissions.deny` | Commands that must never run (sudo, force push, .env read/write) |
| `permissions.ask` | Commands that require confirmation (rm, git push, git merge, etc.) |
| `hooks.PreToolUse` | Before editing a file: warn if editing on main branch |
| `hooks.PostToolUse` | After editing a file: auto-run `ruff check --fix` on .py files |

**Three permission levels**:
```
allow  → Run unconditionally (safe commands)
ask    → Confirm with user every time (risky commands)
deny   → Never run (dangerous commands)
```

### 3. `agents/` - Subagents
**When they are used**: Either auto-selected by Claude or explicitly requested by the user

| Agent | Model | Purpose |
|-------|-------|---------|
| `planner` | opus | Pre-implementation design and impact analysis |
| `code-reviewer` | opus | Code quality and SOLID-principle checks |
| `security-reviewer` | opus | Vulnerability detection (OWASP perspective) |

Subagents perform independent investigation and analysis without polluting the main conversation.
Only a summary of the result is returned to the main conversation.

### 4. `commands/` - Slash Commands
**When they are used**: When the user inputs `/xxx`

Each command has:
- `allowed-tools`: Tools that the command may use
- Procedure: Step-by-step instructions for Claude
- Output format: How results should be displayed

### 5. `rules/` - Always-Applied Rules
**When they are read**: Automatically applied in every conversation (unlike commands/skills, always active)

| Rule | Description |
|------|-------------|
| `coding-style.md` | KISS/YAGNI/DRY, naming conventions, functions under 50 lines, nesting under 3 levels |
| `security.md` | No hardcoded secrets, no eval, SSL verification required |
| `testing.md` | pytest, AAA pattern, 80%+ coverage |
| `git-workflow.md` | Branch strategy, no force push, conflict resolution policy |
| `performance.md` | LLM API timeout, asyncio.gather, rate limit handling |
| `agents.md` | Subagent selection and parallel execution policy |

### 6. `skills/` - Auto-Triggered Skills
**When they are used**: Auto-applied when Claude detects a related task

| Skill | Trigger Condition |
|-------|-------------------|
| `api-patterns` | FastAPI endpoint design / async processing |
| `coding-standards` | Code writing / review / refactoring |
| `frontend-patterns` | React/TypeScript component design |
| `security-review` | Code review / pre-deployment check |

---

## Actual Usage Flows

### Flow 1: Implementing a New Feature

```
1. /plan I want to add a new API authentication endpoint
   → planner agent presents impact scope and implementation steps

2. Ask Claude to implement
   → rules/coding-style.md, rules/security.md are auto-applied
   → skills/api-patterns applies FastAPI patterns
   → hooks auto-run ruff check --fix after editing .py

3. /test
   → Tests are auto-generated (pytest, AAA pattern)
   → Coverage criteria from rules/testing.md are applied

4. /code-review
   → code-reviewer agent runs quality checks

5. /deploy-check
   → Final pre-deployment verification
```

### Flow 2: Bug Fix

```
1. /build-fix paste the error message
   → Identify the root cause and apply the fix

2. /test run
   → Verify existing tests still pass

3. /code-review
   → Review the fix
```

### Flow 3: Refactoring

```
1. /refactor path/to/file.py
   → Present a plan for dead-code removal, function splitting, naming improvements
   → Execute after user approval

2. /test run
   → Verify tests still pass after refactoring
```

### Flow 4: TDD for a New Feature

```
1. /tdd Validation function for user authentication
   → RED: Write a failing test first
   → GREEN: Minimal implementation that passes tests
   → REFACTOR: Improve the code
```

### Flow 5: Log Investigation

```
1. /log-analyze /var/log/app/error.log
   → Classify error patterns, count frequencies, estimate root causes
```

---

## Setting Up for a New Project

### Step 1: Copy the template
```bash
cp -r /path/to/example/.claude /path/to/new-project/.claude
```

### Step 2: Add project-specific information to claude.md
```markdown
## Project-Specific

### Tech Stack
- Backend: FastAPI + PostgreSQL
- Frontend: React + Vite
- Deploy: Docker Compose

### Directory Structure
(Describe the project's structure)

### Project-Specific Commands
(Describe how to build and run the project)
```

### Step 3: Adjust settings.json for the project (optional)
- Only change if additional allow/deny entries are needed
- Customize hooks (e.g., add `eslint --fix` for TypeScript)

---

## Global vs Project Configuration

| Location | Scope | Purpose |
|----------|-------|---------|
| `~/.claude/` | Shared across all projects | Baseline rules, permissions, agents |
| `project/.claude/` | This project only | Project-specific claude.md and extra rules |

**Recommended**: Use this template as the foundation in `~/.claude/`, and only put project-specific information in each project's `.claude/claude.md`.

```bash
# Copy globally (foundation)
cp -r example/.claude/* ~/.claude/

# Put only claude.md in each project
echo "# MyProject\n## Tech Stack\n..." > /path/to/project/.claude/claude.md
```
