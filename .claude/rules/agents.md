# Agent Delegation Rules

## When to Use Subagents

- When security review is needed → `security-reviewer`
- When code review is needed → `code-reviewer`
- When an implementation plan is needed → `planner`

## Principles

- **Minimize tools**: An agent with 5 tools is more efficient than one with 50
- **Limit scope**: One responsibility per agent
- **Model selection**: Use opus for security-related work, sonnet for everything else
- **Parallel execution**: Launch multiple agents concurrently for independent tasks

## Context Window Management

- Use subagents to avoid polluting the main context
- Delegate investigations that require reading many files to subagents
- Return only summarized results to the main context
