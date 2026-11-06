# Releasing Kairn

Pushing a `v*` tag runs `.github/workflows/publish-pypi.yml`: it checks the tag
against both `pyproject.toml` and `__version__` in `src/kairn/__init__.py`,
builds the distributions, tests the built wheel on one Linux / Python 3.12 job,
and publishes to PyPI. The full test matrix runs only in `ci.yml`, so the
commit you tag should have a green `ci.yml` run of its own. The MCP Registry
entry is published separately, by hand, with
`.github/workflows/publish-mcp-registry.yml`.

This file does not yet list every release step (the other version fields, the
`.mcpb` bundle, the registry entry). It holds the one check that has to happen
before any tag.

## Before the tag

Run the quickstart check below on the commit you are about to tag. Every defect
it finds is fixed, or listed as known in `CHANGELOG.md`, before the tag is
pushed.

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
