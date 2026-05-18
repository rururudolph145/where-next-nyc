# Performance Rules

## LLM API Calls

- Set a timeout (recommended: 180 seconds)
- Configure max_retries (recommended: 3-5)
- Execute parallel calls with `asyncio.gather`
- Use exponential backoff for rate limit errors (429)

## Context Window Management

- Do not enable all MCP servers; only enable those used by the project
- Keep the number of active tools under 80
- Minimize the tool set for subagents

## Python

- Use generators when processing large lists
- Parallelize I/O-bound operations with async
- Be mindful of initialization cost when managing singletons via global variables

## Frontend

- Consider virtualization for rendering large lists
- Prevent unnecessary re-renders with `React.memo`, `useMemo`, `useCallback`
- Use pagination when API responses are large

## Kubernetes

- Set resource requests/limits appropriately
- Monitor HPA metrics and adjust scaling thresholds
- Configure pod health checks (readiness/liveness)
