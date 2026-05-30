"""Filesystem invariant tests for the PBI 25a orchestration scaffold.

Verifies that all structural deliverables are present and correct without
requiring Docker, network access, or any external services.

AC coverage:
  AC #1  — app/core/clock.py exists; Clock is a @runtime_checkable Protocol
            with exactly one method: now() -> datetime
            (unit-level coverage: tests/test_clock.py — TestProtocolConformance,
             TestWallClock, TestSimulatedClock)
  AC #2  — WallClock().now() returns UTC-aware datetime within 1 s of system clock
            (unit-level coverage: tests/test_clock.py — TestWallClock)
  AC #3  — SimulatedClock start/advance semantics
            (unit-level coverage: tests/test_clock.py — TestSimulatedClock)
  AC #4  — isinstance checks for both implementations
            (unit-level coverage: tests/test_clock.py — TestProtocolConformance)
  AC #5  — SimulatedClock.advance(negative) raises ValueError
            (unit-level coverage: tests/test_clock.py — TestSimulatedClock)
  AC #6  — SimulatedClock(naive_datetime) raises TypeError
            (unit-level coverage: tests/test_clock.py — TestSimulatedClock)
  AC #7  — apache-airflow not in pyproject.toml
  AC #8  — orchestration/docker-compose.yaml references apache/airflow:3.2.2;
             source URL recorded in orchestration/README.md
  AC #9  — orchestration/README.md names all three planned DAGs + PBI pointers
             and includes cp .env.example .env bootstrap step
  AC #10 — orchestration/.env.example exists, contains AIRFLOW_UID,
             and has a credentials-override warning comment
  AC #11 — dags/, plugins/, config/ each exist with a .gitkeep
  AC #12 — .gitignore contains orchestration/logs and orchestration/.env
             but NOT orchestration/.env.example
  AC #13 — makefile defines airflow-init, airflow-up, airflow-down targets
  AC #14 — tests/test_dag_integrity.py exists and is green
            (covered by: tests/test_dag_integrity.py — TestDagsDirectoryBaseline)
  AC #15 — ruff + pytest CI gates pass on this branch
            (CI-only; no test added here — verified by GitHub Actions)
  Invariant — app/ does not import from orchestration/
"""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent


class TestClockModuleStructure:
    """AC #1 — app/core/clock.py structural invariants.

    Unit-level behaviour tests (WallClock, SimulatedClock, isinstance) live in
    tests/test_clock.py.  These tests lock the module-level structural contract
    so a refactor that renames the Protocol or removes @runtime_checkable is
    caught at the acceptance layer even before import failures surface.
    """

    def test_clock_module_exists(self) -> None:
        """app/core/clock.py must exist.

        AC #1.
        """
        # Arrange
        clock_path = REPO_ROOT / "app" / "core" / "clock.py"

        # Act / Assert
        assert clock_path.exists(), "app/core/clock.py must exist"

    def test_clock_protocol_is_runtime_checkable(self) -> None:
        """Clock must be decorated with @runtime_checkable.

        AC #1 — verifies the decorator is present in the source AST so that
        isinstance(obj, Clock) works at runtime.
        """
        # Arrange
        clock_path = REPO_ROOT / "app" / "core" / "clock.py"
        tree = ast.parse(clock_path.read_text())

        # Act — find the Clock class definition and inspect its decorators
        clock_class_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "Clock":
                clock_class_node = node
                break

        # Assert — class exists
        assert clock_class_node is not None, (
            "No class named 'Clock' found in app/core/clock.py"
        )

        # Assert — @runtime_checkable decorator is present
        decorator_names = []
        for dec in clock_class_node.decorator_list:
            if isinstance(dec, ast.Name):
                decorator_names.append(dec.id)
            elif isinstance(dec, ast.Attribute):
                decorator_names.append(dec.attr)
        assert "runtime_checkable" in decorator_names, (
            "Clock class is not decorated with @runtime_checkable — "
            "isinstance(obj, Clock) will not work at runtime"
        )

    def test_clock_protocol_has_exactly_one_method_now(self) -> None:
        """Clock protocol must define exactly one method: now().

        AC #1 — guards against accidental API surface expansion on the Protocol.
        """
        # Arrange
        clock_path = REPO_ROOT / "app" / "core" / "clock.py"
        tree = ast.parse(clock_path.read_text())

        # Act — collect method names defined on the Clock class body
        clock_class_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "Clock":
                clock_class_node = node
                break

        assert clock_class_node is not None, (
            "No class named 'Clock' found in app/core/clock.py"
        )

        method_names = [
            node.name
            for node in clock_class_node.body
            if isinstance(node, ast.FunctionDef)
        ]

        # Assert
        assert method_names == ["now"], (
            f"Clock protocol must define exactly one method 'now'; "
            f"found: {method_names}"
        )


class TestAirflowNotInPyproject:
    """AC #7 — Airflow must remain container-only."""

    def test_apache_airflow_not_in_pyproject_toml(self) -> None:
        """'apache-airflow' must not appear anywhere in pyproject.toml."""
        # Arrange
        pyproject = (REPO_ROOT / "pyproject.toml").read_text()

        # Act / Assert
        assert "apache-airflow" not in pyproject, (
            "apache-airflow found in pyproject.toml — Airflow is container-only; "
            "do not add it as a Python dependency."
        )


class TestDockerComposeYaml:
    """AC #8 — docker-compose.yaml must reference airflow:3.2.2 and source URL recorded."""

    def test_orchestration_docker_compose_yaml_references_airflow_3_2_2(self) -> None:
        """orchestration/docker-compose.yaml must contain 'apache/airflow:3.2.2'."""
        # Arrange
        compose_path = REPO_ROOT / "orchestration" / "docker-compose.yaml"

        # Act
        content = compose_path.read_text()

        # Assert
        assert compose_path.exists(), "orchestration/docker-compose.yaml must exist"
        assert "apache/airflow:3.2.2" in content, (
            "docker-compose.yaml does not reference apache/airflow:3.2.2"
        )

    def test_orchestration_docker_compose_source_url_recorded_in_readme(
        self,
    ) -> None:
        """The upstream source URL for docker-compose.yaml must be in orchestration/README.md.

        AC #8 — 'The file's source URL is recorded in a comment or in orchestration/README.md'.
        """
        # Arrange
        readme_path = REPO_ROOT / "orchestration" / "README.md"
        source_url = (
            "https://airflow.apache.org/docs/apache-airflow/3.2.2/docker-compose.yaml"
        )

        # Act
        content = readme_path.read_text()

        # Assert
        assert source_url in content, (
            f"orchestration/README.md does not record the source URL "
            f"'{source_url}' — AC #8 requires the origin of docker-compose.yaml "
            "to be documented."
        )


class TestOrchestrationReadme:
    """AC #9 — README must list the three planned DAGs with PBI pointers and bootstrap step."""

    def test_orchestration_readme_lists_three_planned_dags_with_pbi_pointers(
        self,
    ) -> None:
        """orchestration/README.md must name all three DAGs and PBIs 25b/25c/25d."""
        # Arrange
        readme_path = REPO_ROOT / "orchestration" / "README.md"
        content = readme_path.read_text()

        # Act / Assert — DAG names
        assert "refresh_15min_dag" in content, (
            "README.md does not mention refresh_15min_dag"
        )
        assert "train_monthly_dag" in content, (
            "README.md does not mention train_monthly_dag"
        )
        assert "monitor_hourly_dag" in content, (
            "README.md does not mention monitor_hourly_dag"
        )

        # Act / Assert — PBI pointers
        assert "25b" in content, "README.md does not mention PBI 25b"
        assert "25c" in content, "README.md does not mention PBI 25c"
        assert "25d" in content, "README.md does not mention PBI 25d"

    def test_orchestration_readme_includes_cp_env_example_bootstrap_step(
        self,
    ) -> None:
        """orchestration/README.md must include the 'cp .env.example .env' bootstrap step.

        AC #9 — 'The README includes a cp .env.example .env bootstrap step
        before make airflow-init.'
        """
        # Arrange
        readme_path = REPO_ROOT / "orchestration" / "README.md"

        # Act
        content = readme_path.read_text()

        # Assert
        assert "cp" in content and ".env.example" in content, (
            "orchestration/README.md does not include a 'cp .env.example' "
            "bootstrap step — developers need this instruction before "
            "running make airflow-init."
        )


class TestEnvExample:
    """AC #10 — .env.example must exist, contain AIRFLOW_UID, and warn about credentials."""

    def test_orchestration_env_example_exists_and_contains_airflow_uid(self) -> None:
        """orchestration/.env.example must exist and contain 'AIRFLOW_UID'."""
        # Arrange
        env_example = REPO_ROOT / "orchestration" / ".env.example"

        # Act
        content = env_example.read_text()

        # Assert
        assert env_example.exists(), "orchestration/.env.example must exist"
        assert "AIRFLOW_UID" in content, (
            "orchestration/.env.example does not define AIRFLOW_UID"
        )

    def test_orchestration_env_example_contains_credentials_override_warning(
        self,
    ) -> None:
        """orchestration/.env.example must include a comment warning to override credentials.

        AC #10 — 'a comment warning that credentials must be overridden before use'.
        """
        # Arrange
        env_example = REPO_ROOT / "orchestration" / ".env.example"

        # Act
        content = env_example.read_text()

        # Assert — a '#' comment line must contain an override/credentials warning
        warning_present = any(
            line.startswith("#")
            and any(
                kw in line.lower()
                for kw in ("override", "credential", "warning", "production")
            )
            for line in content.splitlines()
        )
        assert warning_present, (
            "orchestration/.env.example has no comment warning about overriding "
            "credentials before production use — AC #10 requires this guard."
        )


class TestGitkeepStubs:
    """AC #11 — dags/, plugins/, config/ must each contain a .gitkeep."""

    def test_orchestration_dags_plugins_config_have_gitkeep(self) -> None:
        """orchestration/dags/, plugins/, config/ each exist with a .gitkeep."""
        # Arrange
        dirs = [
            REPO_ROOT / "orchestration" / "dags",
            REPO_ROOT / "orchestration" / "plugins",
            REPO_ROOT / "orchestration" / "config",
        ]

        # Act / Assert
        for directory in dirs:
            assert directory.exists(), f"{directory} directory must exist"
            assert (directory / ".gitkeep").exists(), (
                f"{directory / '.gitkeep'} must exist"
            )


class TestMakefile:
    """AC #13 — makefile must define all three airflow targets."""

    def test_makefile_defines_airflow_init_airflow_up_airflow_down_targets(
        self,
    ) -> None:
        """makefile must contain target definitions for all three airflow targets."""
        # Arrange
        makefile_path = REPO_ROOT / "makefile"
        content = makefile_path.read_text()

        # Act / Assert
        assert "airflow-init:" in content, "makefile missing 'airflow-init:' target"
        assert "airflow-up:" in content, "makefile missing 'airflow-up:' target"
        assert "airflow-down:" in content, "makefile missing 'airflow-down:' target"

    def test_makefile_recipe_lines_use_tab_indentation(self) -> None:
        """makefile recipe lines must start with a literal tab, not spaces.

        Make is whitespace-sensitive: spaces in front of recipe lines produce
        the classic "missing separator" error. ruff and CI lint do not catch
        this — only running `make` does — so we lock it here.
        """
        # Arrange
        makefile_path = REPO_ROOT / "makefile"
        lines = makefile_path.read_text().splitlines()

        # Act — find every line that follows a target line and is non-empty
        target_names = ("airflow-init:", "airflow-up:", "airflow-down:")
        recipe_lines: list[str] = []
        in_recipe = False
        for line in lines:
            stripped = line.strip()
            if any(line.startswith(name) for name in target_names):
                in_recipe = True
                continue
            if in_recipe and stripped == "":
                in_recipe = False
                continue
            if in_recipe:
                recipe_lines.append(line)

        # Assert — every recipe line begins with a tab character
        assert recipe_lines, "makefile must have at least one recipe line"
        for line in recipe_lines:
            assert line.startswith("\t"), (
                f"makefile recipe line must start with a tab, got: {line!r}"
            )


class TestGitignore:
    """AC #12 — .gitignore must list orchestration/logs and orchestration/.env."""

    def test_gitignore_contains_orchestration_logs_and_dot_env_but_not_dot_env_example(
        self,
    ) -> None:
        """.gitignore must have orchestration/logs* and orchestration/.env entries.

        orchestration/.env.example must NOT be in .gitignore.
        """
        # Arrange
        gitignore_path = REPO_ROOT / ".gitignore"
        content = gitignore_path.read_text()

        # Assert — orchestration/logs must be gitignored
        assert "orchestration/logs" in content, (
            ".gitignore does not contain orchestration/logs"
        )

        # Assert — orchestration/.env must be gitignored explicitly
        assert "orchestration/.env" in content, (
            ".gitignore does not contain orchestration/.env"
        )

        # Assert — .env.example must NOT be gitignored
        assert "orchestration/.env.example" not in content, (
            ".gitignore must not include orchestration/.env.example — "
            "the template file must be tracked."
        )


class TestAppDoesNotImportOrchestration:
    """Dependency direction invariant: app/ must never import from orchestration/."""

    def test_app_does_not_import_from_orchestration(self) -> None:
        """No .py file under app/ may contain 'from orchestration' or 'import orchestration'."""
        # Arrange
        app_dir = REPO_ROOT / "app"
        violations: list[str] = []

        # Act
        for py_file in app_dir.rglob("*.py"):
            file_text = py_file.read_text()
            if "from orchestration" in file_text or "import orchestration" in file_text:
                violations.append(str(py_file))

        # Assert
        assert violations == [], (
            "The following app/ files import from orchestration/ — "
            "the dependency direction must be one-way (orchestration → app, never app → orchestration): "
            f"{violations}"
        )
