# Rules

- Do not preserve backward compatibility. Remove obsolete paths instead of adding compatibility layers, fallbacks, or migrations.
- Choose the simplest implementation that fully meets the current requirements. Avoid speculative abstractions, configuration, and indirection.
- Grow the system in layers. Start from the smallest version that works end to end, and add each new capability on top of a product that already works. Never trade a working product for unfinished complexity.
- Keep components modular and concerns clearly separated.
- Prefer established, well-maintained libraries when they reduce overall complexity or improve reliability. Do not reimplement common functionality without a clear reason.
- Lean on the dependencies already in the project before writing your own implementation or adding packages. Do not assume a library lacks a capability without checking its documentation and types.
- Make architectural decisions for the long term. Do not accept a stopgap that only works for now and is meant to be replaced later.

# Environment

- Run Python through the checkout-local uv environment (`.venv`), never the
  system python or another checkout's venv. Create it once with
  `uv venv .venv && uv pip install pytest` — runs are stdlib-only, only the
  tests need pytest.
- Tests import the package as `eval`, so invoke them from the directory
  where this checkout is importable under that name:
  `eval/.venv/bin/python -m pytest eval/tests -q`.

# Commit messages

Use the following format for commit messages:

- `ci: description` for all testing related changes, and changes to github workflows etc.
- `dev: description` for development related changes, including updates to the cursor or claude rules
- `fix(component): description` for bug fixes
- `feat(component): description` for new features
- `enh(component): description` for enhancements
- `docs: description` for documentation
- `ref(component): description` for refactoring
- `chore: description` for maintenance tasks (pre-commit hooks, imports, etc.)

Generally, the description should focus on the intent of the changes, not the implementation details.

## Style notes

Do **NOT** add "Co-authored-by: Cursor" lines to the commit message or to the trailer.

