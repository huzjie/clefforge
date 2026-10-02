"""Backend registry."""

_REGISTRY = {}


def register_backend(name, cls=None):
    def _wrap(c):
        _REGISTRY[name] = c
        c.name = name
        return c
    if cls is not None:
        return _wrap(cls)
    return _wrap


def list_backends():
    return sorted(_REGISTRY.keys())


def get_backend(name):
    if name not in _REGISTRY:
        raise KeyError(f"unknown backend '{name}' (available: {', '.join(list_backends())})")
    return _REGISTRY[name]
