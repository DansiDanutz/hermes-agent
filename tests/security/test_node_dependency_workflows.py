from pathlib import Path


ROOT = Path(__file__).parents[2]
TESTS_WORKFLOW = ROOT / ".github" / "workflows" / "tests.yml"
OSV_WORKFLOW = ROOT / ".github" / "workflows" / "osv-scanner.yml"


def _production_lockfiles() -> set[str]:
    return {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("package-lock.json")
        if "node_modules" not in path.parts
    }


def test_every_node_lockfile_has_blocking_audit_and_osv_coverage() -> None:
    """Every committed Node graph must retain both blocking and advisory scans."""
    tests_workflow = TESTS_WORKFLOW.read_text(encoding="utf-8")
    osv_workflow = OSV_WORKFLOW.read_text(encoding="utf-8")

    assert "npm audit --omit=dev --audit-level=${{ matrix.audit-level }}" in tests_workflow

    for lockfile in _production_lockfiles():
        directory = str(Path(lockfile).parent).replace("\\", "/")
        matrix_entry = "." if directory == "." else directory

        assert f"          - directory: {matrix_entry}\n" in tests_workflow
        assert f"--lockfile={lockfile}" in osv_workflow
