---
name: build-fix
description: Investigates and fixes build or runtime errors. Identifies the root cause from error messages.
allowed-tools: Read, Glob, Grep, Bash, Edit, Write
---

# Build Error Resolver

Resolve build or runtime errors.

## Procedure

1. **Reproduce the error**: If an error message is provided in the arguments, analyze it; otherwise run:
   - Python: `python -m py_compile <file>` or ruff check
   - TypeScript: `npx tsc --noEmit`
   - Build: `npm run build`

2. **Identify the root cause**:
   - Extract the file name and line number from the error message
   - Trace the stack trace in reverse order
   - Determine whether it is a dependency issue or a code issue

3. **Apply the fix**:
   - Make the minimal change needed
   - Check for side effects
   - Re-run the build to verify the fix

4. **Report**:
   - Explanation of the cause
   - Description of the applied fix
   - Suggestions to prevent recurrence

Error content: $ARGUMENTS
