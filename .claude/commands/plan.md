---
name: plan
description: Plans the implementation of a feature. Clarifies requirements, impact, and implementation steps.
allowed-tools: Read, Glob, Grep, Task
---

# Implementation Planner

For the given feature requirement, create an implementation plan using the following procedure.

## Procedure

1. **Requirement analysis**: Organize the user's request and clarify functional and non-functional requirements
2. **Impact investigation**: Identify files that need changes and components that will be affected
3. **Existing patterns**: Investigate similar implementations in the project to maintain consistency
4. **Implementation steps**: Output concrete work steps as a task list
5. **Risk assessment**: Evaluate breaking changes, backward compatibility, and performance impact

## Output Format

```markdown
## Overview
[1-2 sentences describing what will be implemented]

## Affected Files
- `path/to/file.py` - [change description]

## Implementation Steps
1. [ ] Step 1
2. [ ] Step 2

## Risks & Considerations
- [risk item]
```

User request: $ARGUMENTS
