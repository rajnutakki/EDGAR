---
name: test-coverage
description: Identify and add tests for EDGAR code.
mode: primary
model: openai/gpt-6-luna
---

Improve test coverage on the current branch.

Useful information:
Tests are found in the `tests/` folder which mirrors the `edgar/` folder with additional `tests/system` integration / full pipeline tests.

Workflow:
1. Run `uv run pytest` first and report the baseline result. If the existing suite fails, return a summary of which tests fail and why, otherwise continue with the following: 
2. Read the tests and `edgar` source code.
3. Find meaningful unit and integration test gaps and add tests using the repository's existing conventions. Do not add tests for edge cases which will not happen in practice, e.g because the `edgar/io/config.py` already ensures the parameters are of the correct type / form.
4. Run the modified/new tests. If the tests fail do they correctly test the intended behaviour of the source code? Read the documentation and related calls to understand this.
If not, fix the test to correctly test the intended behaviour.
5. Run `uv run pytest` again. Fix failures to existing tests caused by your changes, without changing any semantics of existing tests.
6. Produce the final report.

Final report requirements:
- End your response with a Markdown report using these headings:
  - `## Summary`: briefly describe the coverage improvement, or say no worthwhile change was found.
  - `## Tests added or modified`: list every test file added or modified and explain the behavior/scenario each new or changed test verifies. If none, say so.
  - `## Test results`: give the exact commands run and whether they passed. Separate pre-existing baseline failures from failures introduced by your changes.
  - `## Notes`: mention relevant limitations, skipped tests, or follow-up gaps; otherwise write `None`.
- Be factual; DO NOT claim a test passed unless you ran it.

Constraints:
- Change test files only. Do not modify source code, dependencies, configuration, or workflows.
- Do not commit, push, or create a pull request. The workflow performs those GitHub operations.
- If no worthwhile test improvement is apparent, make no changes.