# PBI 3 — Heatmap API Implementation Checklist

## Requirements

- **Endpoint:** `GET /heatmap`
- **Input:** None (reads latest predictions parquet)
- **Output:** JSON with `generated_at` + list of `{ pulocationid, zone_name, predicted_trip_count }`
- **Acceptance Criteria:**
  - Response < 200ms
  - Returns valid zone predictions
  - Serves from predictions store (`data/predictions/demand_predictions.parquet`)

## Implementation Steps

- [x] 1. Add `serving` section to `config.yml` with predictions path and zone shapefile path
- [x] 2. Implement `GET /heatmap` endpoint in `src/serving/app.py`
  - Load predictions parquet (latest hour or all hours, grouped by zone)
  - Join `pulocationid` → `zone_name` via zone shapefile lookup
  - Return JSON: `{ generated_at, predictions: [{ pulocationid, zone_name, predicted_trip_count }] }`
  - Cache the predictions DataFrame at startup (not on every request) for < 200ms response
- [x] 3. Add optional `?hour=YYYY-MM-DDTHH:00:00` query param to filter by specific hour (default: latest)
- [x] 4. Add `uvicorn` to `pyproject.toml` dependencies (currently missing)
- [x] 5. Write tests in `tests/test_serving.py`
  - Test `/` health check
  - Test `GET /heatmap` returns valid schema
  - Test `GET /heatmap?hour=...` filters correctly
  - Test response when predictions file is missing
- [x] 6. Run `uv run mypy src/serving/ --strict` — all type checks pass
- [x] 7. Run `uv run ruff check src/serving/` — all lint rules pass
- [x] 8. Run `uv run ruff format --check src/serving/` — formatting consistent
- [x] 9. Update PBI tracker to mark PBI 3 as done
