from pathlib import Path


WORKFLOW = Path(__file__).parents[2] / ".github" / "workflows" / "nix-lockfile-fix.yml"


def test_privileged_pr_repair_rejects_fork_before_checkout() -> None:
    """Fork-controlled code must never receive the repair job's secrets."""
    workflow = WORKFLOW.read_text()
    fork_guard = "pr.head.repo.full_name !== `${context.repo.owner}/${context.repo.repo}`"
    privileged_checkout = "repository: ${{ steps.resolve.outputs.owner }}/${{ steps.resolve.outputs.repo }}"

    assert fork_guard in workflow
    assert privileged_checkout in workflow
    assert workflow.index(fork_guard) < workflow.index(privileged_checkout)


def test_main_auto_fix_skips_safely_without_github_app_credentials() -> None:
    """Missing optional write-back credentials must not make main CI fail."""
    workflow = WORKFLOW.read_text()
    credential_check = "- name: Check automation credentials"
    token_step = "- name: Generate GitHub App token"
    guard = "if: steps.app-credentials.outputs.configured == 'true'"

    assert credential_check in workflow
    assert "APP_ID: ${{ secrets.APP_ID }}" in workflow
    assert "APP_PRIVATE_KEY: ${{ secrets.APP_PRIVATE_KEY }}" in workflow
    assert "Nix auto-fix skipped" in workflow
    assert workflow.index(credential_check) < workflow.index(token_step)

    auto_fix = workflow.split("  # ── PR fix", maxsplit=1)[0]
    privileged_steps = auto_fix.split(token_step, maxsplit=1)[1]
    assert privileged_steps.count(guard) == 5
