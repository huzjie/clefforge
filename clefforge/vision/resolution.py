"""Arbitrary-resolution policy: dynamic tiling vs fixed grid.

Clef accepts arbitrary-resolution input; small images use a single fixed grid,
large ones are split into tiles (each tile its own patch grid) and their tokens
concatenated. This module computes the grid/tiling decision deterministically.
"""


def resolve_resolution(width, height, max_patches=512, patch_size=16):
    pw = max(1, width // patch_size)
    ph = max(1, height // patch_size)
    total = pw * ph
    if total <= max_patches:
        return {"mode": "single", "pw": pw, "ph": ph, "n_patches": total}
    # tiling: split into k tiles along the longer axis until under budget
    k = 1
    while (pw * ph) // k > max_patches:
        k += 1
    return {"mode": "tiled", "pw": pw, "ph": ph, "n_patches": total, "tiles": k}
