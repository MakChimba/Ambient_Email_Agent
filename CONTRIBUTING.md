# Contributing Guide

This project favors live Gemini verification whenever credentials are available. Use the offline toggles (`EMAIL_ASSISTANT_EVAL_MODE=1`, etc.) only when a deterministic run is strictly required.

## Post-merge Housekeeping
- After a pull request merges, sync your local `main` (`git fetch && git checkout main && git pull`) before starting the next branch.
- Remove the merged feature branch in both remotes and your local clone to avoid stale configurations:
  - Remote: `git push origin --delete <branch>`
  - Local: `git branch -d <branch>`
- If you want this automated, enable GitHub’s **Automatically delete head branches** repository setting or add a cleanup workflow (e.g., [`peter-evans/delete-merged-branch`](https://github.com/peter-evans/delete-merged-branch)).

## CodeRabbit Reviews & Auto-Merge
- CodeRabbit approvals now trigger an automation that enables GitHub’s squash auto-merge for the pull request.
- Auto-merge will complete once the configured status checks pass; no manual intervention is required unless GitHub blocks the merge (e.g., conflicts, missing status checks).
- CodeRabbit itself cannot delete branches or merge PRs; the workflow handles auto-merge enabling, while GitHub’s branch settings govern the rest.

## Testing Expectations
- Default to running the live Gemini suites (`pytest tests/test_live_smoke.py`, etc.). Offline paths should be treated as fallbacks and documented in the PR description when used.
