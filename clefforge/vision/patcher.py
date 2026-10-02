"""Image -> patches (deterministic feature synthesis for the mock backend).

The real Clef patches the image into 16x16 tiles; here we synthesize per-patch
features from a stable hash of (image_id, patch_index) so the whole pipeline is
reproducible without shipping real pixel decoding. Swap in real patch embeddings
by overriding `Patcher.patch_features`.
"""
from ..utils.stable import stable_float, stable_vector


class Patcher:
    def __init__(self, patch_size=16, hidden=128):
        self.patch_size = patch_size
        self.hidden = hidden

    def patch_features(self, image_id, n_patches):
        return [stable_vector(f"patch:{image_id}:{i}", self.hidden) for i in range(n_patches)]

    def num_patches(self, width, height):
        return max(1, (width // self.patch_size) * (height // self.patch_size))
