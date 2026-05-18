---
name: log-analyze
description: Analyzes application logs to identify error patterns, frequencies, and root causes.
allowed-tools: Read, Grep, Bash
---

# Log Analyzer

Analyze the specified log file or log output.

## Procedure

1. **Obtain logs**: If a file path or log text is provided in the arguments, use it

2. **Classify errors**:
   - Extract ERROR/CRITICAL level logs
   - Group by error type
   - Count occurrences of each error

3. **Pattern analysis**:
   - Distribution of errors by time of day
   - Whether they are concentrated on specific users
   - Detect cascading errors (one error triggering others)

4. **Root cause estimation**:
   - Analyze stack traces
   - Use surrounding logs to understand context
   - Locate the affected code

## Output Format

```markdown
## Error Summary
| Error Type | Count | Severity |
|-----------|-------|----------|
| ...       | ...   | ...      |

## Detailed Analysis
### Error 1: [error name]
- Occurrences: N
- Cause: [description]
- Affected code: `file:line`
- Recommended action: [how to address]

## Recommended Actions (by priority)
1. [Top-priority action]
```

Target: $ARGUMENTS
