# Releasing Kairn

The upload itself is automatic. Pushing a `v*` tag runs
`.github/workflows/publish-pypi.yml`, which checks that the tag matches the
package version, tests, builds and publishes through PyPI Trusted Publishing.
The MCP Registry entry is published by hand with
`.github/workflows/publish-mcp-registry.yml`, after the release asset exists and
has been checked. This file is what happens before the tag.

## Before the tag

1. CI is green on `main` (`.github/workflows/ci.yml`).
2. The version in `pyproject.toml` is the version you are about to tag.
3. `CHANGELOG.md` moves the `Unreleased` entries under that version.
4. The quickstart check below found no open defect, or every defect it found is
   fixed or listed in `CHANGELOG.md` as known.

## Quickstart check

Run it once per release, as a reader who has never seen Kairn: a fresh virtual
environment, the README open, nothing else.

1. **Name the success moment first.** Before running anything, write down the
   one thing a new user should see when the quickstart worked, for example
   `kairn status ~/brain` printing its JSON stats block.
2. **Follow the README exactly**, command by command, in its order. Fix nothing
   quietly: a command you had to change before it ran is a finding.
3. **Report two kinds of defect only:**
   - *doc vs code*: a command, flag, default, file name or step order the README
     states that the code does not have (check it against `src/kairn/cli.py`);
   - *output vs effect*: an output or result the README promises that a real run
     does not produce.

   Leave out everything else: missing prerequisites, environment warnings, slow
   steps, wording. They are not release blockers and they bury the real ones.
4. **Keep the report short:** one line per defect (README line, what it says,
   what happens, the command that shows it). The whole report is no longer than
   about one and a half times a plain "it worked" or "it did not" answer.

`tests/test_readme_matches_cli.py` already checks the CLI list, the flags and
several examples on every push. This check covers what that test cannot see:
the path a new reader actually walks.
