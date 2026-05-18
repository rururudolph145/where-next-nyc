---
name: api-patterns
description: Design patterns for FastAPI backends. Apply for endpoint design, async processing, and external API calls.
allowed-tools: Read, Grep, Glob
---

# API Patterns

## Endpoint Design

### URL Naming Conventions
- Resource names use plural form: `/api/ideas`, `/api/teams`
- Nesting up to 2 levels: `/api/teams/{id}/members`
- Use verbs for actions: `/api/ideas/generate`, `/api/discussion/start`

### Request / Response
```python
class CreateIdeaRequest(BaseModel):
    theme: str = Field(..., description="Theme", min_length=1)
    num_ideas: int = Field(default=5, ge=1, le=20)

class ApiResponse(BaseModel):
    status: str
    data: dict | None = None
    error: str | None = None
```

### Error Handling Pattern
```python
@app.post("/api/resource")
async def create_resource(request: ResourceRequest):
    try:
        result = await process(request)
        return {"status": "success", "data": result}
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return JSONResponse(status_code=500, content={"error": "Internal server error"})
```

## Async Processing

### Parallel Execution
```python
results = await asyncio.gather(
    task1(), task2(), task3(),
    return_exceptions=True
)
```

### Timeout Control
```python
try:
    result = await asyncio.wait_for(long_task(), timeout=180.0)
except asyncio.TimeoutError:
    logger.warning("Task timed out")
```

## External API Calls

### httpx (async, with SSL verification)
```python
async with httpx.AsyncClient(timeout=30.0, verify=ssl_context) as client:
    response = await client.post(url, json=payload, headers=headers)
    response.raise_for_status()
```

### Retry (tenacity)
```python
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
async def call_external_api():
    ...
```
