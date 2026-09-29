"""Result directory for an example script.

The folder sits next to the script. Its name is the script stem plus the key
parameters of the run, in the order they are passed:

    fresnel_slab.py with epsr=4, lam=46, dx=0.5
    -> fresnel_slab_epsr4_lam46_dx0.5/
"""

from __future__ import annotations

from pathlib import Path


def output_dir(script_file: str, **parameters: float | int | str) -> Path:
    """Create ``<script>_<key><value>_...`` beside ``script_file``."""
    script = Path(script_file).resolve()
    if not parameters:
        raise ValueError("output_dir needs the key parameters of this run")
    label = "_".join(f"{key}{_token(value)}" for key, value in parameters.items())
    folder = script.parent / f"{script.stem}_{label}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _token(value: float | int | str) -> str:
    if isinstance(value, float):
        return format(value, ".12g")
    return str(value)
