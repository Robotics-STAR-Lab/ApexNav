"""
Path utilities for resolving file paths.

PROJECT_ROOT: The ApexNav project root.
WORKSPACE_ROOT: The parent directory of PROJECT_ROOT.
"""

from pathlib import Path
from typing import Optional, Union


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent


def resolve_existing_path(
    path: Optional[Union[str, Path]], *fallbacks: Union[str, Path]
) -> str:
    candidates = []
    seen = set()

    for raw_path in (path, *fallbacks):
        if raw_path in (None, ""):
            continue

        candidate = Path(raw_path).expanduser()
        raw_candidates = [candidate]
        if not candidate.is_absolute():
            raw_candidates = [
                Path.cwd() / candidate,
                PROJECT_ROOT / candidate,
                WORKSPACE_ROOT / candidate,
            ]

        for item in raw_candidates:
            resolved = item.resolve(strict=False)
            resolved_str = str(resolved)
            if resolved_str in seen:
                continue
            seen.add(resolved_str)
            candidates.append(resolved)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    tried = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"Could not find required path. Tried:\n{tried}")
