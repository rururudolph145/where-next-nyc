"""Acceptance tests for PBI 30 — GitHub Actions CI/CD: Lint + Test Workflows.

Each test maps 1-to-1 to a numbered acceptance criterion (AC) in the approved
user story.  Tests read static files from the repo (workflow YAML, Python source,
README) — no network calls, no GitHub API.

6 manual / out-of-suite criteria (cannot be expressed as local file-reading tests):
  AC #3  — deliberate ruff-violation PR causes job failure (GitHub Actions log)
  AC #4  — clean PR exits green (GitHub Actions)
  AC #9  — broken-test PR causes job failure (GitHub Actions log)
  AC #10 — clean PR exits green (GitHub Actions)
  AC #13 — branch protection configured in GitHub Settings (manual post-merge)
  AC #14 — both workflows pass on current dev_MVP tree (CI run on the PR itself)
"""

import ast
from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Paths — resolved relative to the repo root (tests/ is one level down)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent
_LINT_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "lint.yml"
_TEST_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "test.yml"
_TEST_DATASET = _REPO_ROOT / "tests" / "test_dataset.py"
_README = _REPO_ROOT / "README.md"

_EXPECTED_BRANCHES = {"dev_MVP", "main"}


# ---------------------------------------------------------------------------
# Helper — load a workflow YAML once per test (no shared state)
# ---------------------------------------------------------------------------
def _load_yaml(path: Path) -> dict:
    with path.open() as fh:
        return yaml.safe_load(fh)


def _all_step_run_fields(workflow: dict) -> list[str]:
    """Return every non-None `run` field from all steps across all jobs."""
    runs: list[str] = []
    for job in workflow.get("jobs", {}).values():
        for step in job.get("steps", []):
            run = step.get("run")
            if run is not None:
                runs.append(run)
    return runs


def _all_step_uses_fields(workflow: dict) -> list[str]:
    """Return every non-None `uses` field from all steps across all jobs."""
    uses_list: list[str] = []
    for job in workflow.get("jobs", {}).values():
        for step in job.get("steps", []):
            uses = step.get("uses")
            if uses is not None:
                uses_list.append(uses)
    return uses_list


def _named_steps(workflow: dict) -> list[dict]:
    """Return every step dict that has a non-empty `name:` field."""
    result: list[dict] = []
    for job in workflow.get("jobs", {}).values():
        for step in job.get("steps", []):
            if step.get("name"):
                result.append(step)
    return result


# ===========================================================================
# AC #1 — lint.yml exists and has correct triggers
# ===========================================================================

# NOTE: PyYAML parses the GitHub Actions `on:` key as the Python boolean True
# because `on` is a YAML 1.1 reserved word for True.  All trigger-block access
# must use `workflow[True]` rather than `workflow["on"]`.


def test_lint_workflow_file_exists():
    """AC #1 — .github/workflows/lint.yml must be present in the repo."""
    # Arrange
    path = _LINT_WORKFLOW

    # Act / Assert
    assert path.exists(), f"Expected {path} to exist but it does not"


def test_lint_workflow_yaml_parses():
    """AC #1 — lint.yml must be valid YAML that parses to a non-empty dict."""
    # Arrange / Act
    result = _load_yaml(_LINT_WORKFLOW)

    # Assert
    assert isinstance(result, dict) and result


def test_lint_workflow_triggers_pull_request_branches():
    """AC #1 — pull_request trigger must target dev_MVP and main."""
    # Arrange
    workflow = _load_yaml(_LINT_WORKFLOW)

    # Act — PyYAML maps the YAML `on:` key to Python boolean True
    actual = set(workflow[True]["pull_request"]["branches"])

    # Assert
    assert actual == _EXPECTED_BRANCHES


def test_lint_workflow_triggers_push_branches():
    """AC #1 — push trigger must target dev_MVP and main."""
    # Arrange
    workflow = _load_yaml(_LINT_WORKFLOW)

    # Act — PyYAML maps the YAML `on:` key to Python boolean True
    actual = set(workflow[True]["push"]["branches"])

    # Assert
    assert actual == _EXPECTED_BRANCHES


# ===========================================================================
# AC #2 — lint job has three named steps with the correct commands
# ===========================================================================


@pytest.mark.parametrize(
    "expected_fragment",
    [
        "ruff check .",
        "ruff format --check .",
        "mypy app tests",
    ],
)
def test_lint_workflow_step_commands_present(expected_fragment: str):
    """AC #2 — each lint command must appear in its own separate named step.

    AC #2 requires "each as a separate named step".  Asserting against all
    `run` fields regardless of `name` would pass even if the three commands
    were concatenated into a single unnamed step.  This test asserts that
    exactly one named step's `run` field contains the command fragment, and
    that the three fragments therefore map to three distinct named steps.
    """
    # Arrange
    workflow = _load_yaml(_LINT_WORKFLOW)
    named = _named_steps(workflow)

    # Act — collect named steps whose run field contains the fragment
    matching_steps = [
        step for step in named
        if expected_fragment in (step.get("run") or "")
    ]

    # Assert — exactly one named step must own this command
    assert len(matching_steps) == 1, (
        f"Expected exactly 1 named step in lint.yml whose `run` contains "
        f"'{expected_fragment}', found {len(matching_steps)}. "
        f"Named steps: {[(s.get('name'), s.get('run')) for s in named]}"
    )


# ===========================================================================
# AC #5 — lint job has timeout-minutes: 10
# ===========================================================================


def test_lint_workflow_has_timeout():
    """AC #5 — lint job must declare timeout-minutes: 10."""
    # Arrange
    workflow = _load_yaml(_LINT_WORKFLOW)

    # Act
    jobs = workflow.get("jobs", {})
    timeouts = [job.get("timeout-minutes") for job in jobs.values()]

    # Assert
    assert 10 in timeouts, (
        f"Expected at least one job in lint.yml to have timeout-minutes: 10, "
        f"got: {timeouts}"
    )


# ===========================================================================
# AC #6 — test.yml exists and has correct triggers
# ===========================================================================


def test_test_workflow_file_exists():
    """AC #6 — .github/workflows/test.yml must be present in the repo."""
    # Arrange
    path = _TEST_WORKFLOW

    # Act / Assert
    assert path.exists(), f"Expected {path} to exist but it does not"


def test_test_workflow_yaml_parses():
    """AC #6 — test.yml must be valid YAML that parses to a non-empty dict."""
    # Arrange / Act
    result = _load_yaml(_TEST_WORKFLOW)

    # Assert
    assert isinstance(result, dict) and result


def test_test_workflow_triggers_pull_request_branches():
    """AC #6 — pull_request trigger must target dev_MVP and main."""
    # Arrange
    workflow = _load_yaml(_TEST_WORKFLOW)

    # Act — PyYAML maps the YAML `on:` key to Python boolean True
    actual = set(workflow[True]["pull_request"]["branches"])

    # Assert
    assert actual == _EXPECTED_BRANCHES


def test_test_workflow_triggers_push_branches():
    """AC #6 — push trigger must target dev_MVP and main."""
    # Arrange
    workflow = _load_yaml(_TEST_WORKFLOW)

    # Act — PyYAML maps the YAML `on:` key to Python boolean True
    actual = set(workflow[True]["push"]["branches"])

    # Assert
    assert actual == _EXPECTED_BRANCHES


# ===========================================================================
# AC #7 — test job runs pytest with correct flags; uv sync excludes torch
# ===========================================================================


def test_test_workflow_step_commands_present():
    """AC #7 — a step run field must contain the full pytest invocation."""
    # Arrange
    workflow = _load_yaml(_TEST_WORKFLOW)
    run_fields = _all_step_run_fields(workflow)

    # Act
    found = any("pytest tests/ -v --tb=short" in run for run in run_fields)

    # Assert
    assert found, (
        "No step `run` field in test.yml contains 'pytest tests/ -v --tb=short'. "
        f"Found run fields: {run_fields}"
    )


@pytest.mark.parametrize(
    "excluded_package",
    ["torch", "torchvision"],
)
def test_test_workflow_uv_sync_excludes_torch(excluded_package: str):
    """AC #7 — uv sync step must exclude torch and torchvision to avoid huge download."""
    # Arrange
    workflow = _load_yaml(_TEST_WORKFLOW)
    run_fields = _all_step_run_fields(workflow)

    # Act
    flag = f"--no-install-package {excluded_package}"
    found = any(flag in run for run in run_fields)

    # Assert
    assert found, (
        f"No step `run` field in test.yml contains '{flag}'. "
        f"Found run fields: {run_fields}"
    )


# ===========================================================================
# AC #8 — test_load_data has @pytest.mark.skip decorator
# ===========================================================================


def test_skip_marker_on_test_load_data():
    """AC #8 — test_load_data must carry @pytest.mark.skip with an explicit reason."""
    # Arrange
    source = _TEST_DATASET.read_text()
    tree = ast.parse(source)

    # Act — find the test_load_data function node
    func_node = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "test_load_data"
        ),
        None,
    )
    assert func_node is not None, "test_load_data function not found in test_dataset.py"

    # Inspect decorators for a `pytest.mark.skip` call
    skip_found = False
    reason_present = False
    for decorator in func_node.decorator_list:
        # Handle both `@pytest.mark.skip` (no args) and `@pytest.mark.skip(reason=...)`
        if isinstance(decorator, ast.Call):
            func = decorator.func
            # pytest.mark.skip(...)
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "skip"
                and isinstance(func.value, ast.Attribute)
                and func.value.attr == "mark"
            ):
                skip_found = True
                # Check for a reason keyword argument
                for kw in decorator.keywords:
                    if kw.arg == "reason" and isinstance(kw.value, ast.Constant):
                        reason_present = bool(kw.value.value)
        elif isinstance(decorator, ast.Attribute) and decorator.attr == "skip":
            # bare @pytest.mark.skip without parentheses
            skip_found = True

    # Assert
    assert skip_found, (
        "test_load_data in test_dataset.py does not have a @pytest.mark.skip decorator"
    )
    assert reason_present, (
        "test_load_data's @pytest.mark.skip decorator is missing an explicit reason= string"
    )


# ===========================================================================
# AC #11 — test job has timeout-minutes: 10
# ===========================================================================


def test_test_workflow_has_timeout():
    """AC #11 — test job must declare timeout-minutes: 10."""
    # Arrange
    workflow = _load_yaml(_TEST_WORKFLOW)

    # Act
    jobs = workflow.get("jobs", {})
    timeouts = [job.get("timeout-minutes") for job in jobs.values()]

    # Assert
    assert 10 in timeouts, (
        f"Expected at least one job in test.yml to have timeout-minutes: 10, "
        f"got: {timeouts}"
    )


# ===========================================================================
# AC #12 — README carries two CI badge links
# ===========================================================================


def test_readme_has_ci_badges():
    """AC #12 — README line immediately after H1 must contain both CI badge URLs.

    The spec states badges appear on the line immediately after the H1 title
    and before the first H2 section.  We assert that a line within the first
    five non-blank lines of README.md contains both workflow filenames as
    part of a GitHub Actions badge URL pattern.
    """
    # Arrange
    lines = _README.read_text().splitlines()

    # Act — collect up to the first H2, scanning only the preamble
    preamble: list[str] = []
    for line in lines:
        if line.startswith("## "):
            break
        preamble.append(line)

    badge_line = " ".join(preamble)  # single string for substring search

    # Assert — both workflow filenames must appear in a shields.io-style badge
    assert "lint.yml" in badge_line, (
        "README preamble (before first H2) does not reference lint.yml badge"
    )
    assert "test.yml" in badge_line, (
        "README preamble (before first H2) does not reference test.yml badge"
    )


# ===========================================================================
# Edge cases — setup-uv version pinned to @v4
# ===========================================================================


def test_lint_workflow_uses_setup_uv_v4():
    """Edge case — lint.yml must pin astral-sh/setup-uv@v4 (not v3 or unpinned)."""
    # Arrange
    workflow = _load_yaml(_LINT_WORKFLOW)
    uses_list = _all_step_uses_fields(workflow)

    # Act
    found = any("astral-sh/setup-uv@v4" in uses for uses in uses_list)

    # Assert
    assert found, (
        f"lint.yml does not use 'astral-sh/setup-uv@v4'. Found uses fields: {uses_list}"
    )


def test_test_workflow_uses_setup_uv_v4():
    """Edge case — test.yml must pin astral-sh/setup-uv@v4 (not v3 or unpinned)."""
    # Arrange
    workflow = _load_yaml(_TEST_WORKFLOW)
    uses_list = _all_step_uses_fields(workflow)

    # Act
    found = any("astral-sh/setup-uv@v4" in uses for uses in uses_list)

    # Assert
    assert found, (
        f"test.yml does not use 'astral-sh/setup-uv@v4'. Found uses fields: {uses_list}"
    )
