from pathlib import Path

import yaml


ROOT = Path(__file__).parents[2]
TESTS_WORKFLOW = ROOT / ".github" / "workflows" / "tests.yml"
OSV_WORKFLOW = ROOT / ".github" / "workflows" / "osv-scanner.yml"

EXPECTED_NODE_AUDIT_POLICY = {
    ".": {"audit-level": "high", "engine-strict": True, "node-version": 22},
    "scripts/whatsapp-bridge": {
        "audit-level": "high",
        "engine-strict": True,
        "node-version": 20,
    },
    "ui-tui": {"audit-level": "high", "engine-strict": True, "node-version": 20},
    "ui-tui/packages/hermes-ink": {
        "audit-level": "high",
        "engine-strict": False,
        "node-version": 20,
    },
    "web": {"audit-level": "critical", "engine-strict": True, "node-version": 20},
    "website": {"audit-level": "high", "engine-strict": True, "node-version": 20},
}


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
    workflow = yaml.safe_load(tests_workflow)
    entries = workflow["jobs"]["node-security"]["strategy"]["matrix"]["include"]
    configured_policy = {
        entry["directory"]: {
            "audit-level": entry["audit-level"],
            "engine-strict": entry["engine-strict"],
            "node-version": entry["node-version"],
        }
        for entry in entries
    }

    assert "npm audit --omit=dev --audit-level=${{ matrix.audit-level }}" in tests_workflow
    assert "node-version: ${{ matrix.node-version }}" in tests_workflow
    assert "NPM_CONFIG_ENGINE_STRICT: ${{ matrix.engine-strict }}" in tests_workflow
    assert configured_policy == EXPECTED_NODE_AUDIT_POLICY

    for lockfile in _production_lockfiles():
        directory = str(Path(lockfile).parent).replace("\\", "/")
        matrix_entry = "." if directory == "." else directory

        assert matrix_entry in configured_policy
        assert f"--lockfile={lockfile}" in osv_workflow
