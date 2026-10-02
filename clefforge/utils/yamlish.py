"""Minimal YAML-subset parser (zero-dependency fallback).

Only the subset used by config files is supported: flat mappings, nested mappings,
lists of scalars, integers / floats / booleans / strings. This keeps the CLI able
to read .yaml configs on a managed Python that does not ship PyYAML.
"""
import re


def parse(text):
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    root = {}
    stack = [( -1, root, None)]
    for ln in lines:
        stripped = ln.rstrip()
        indent = len(ln) - len(ln.lstrip(" "))
        content = stripped.strip()
        if ":" not in content:
            continue
        key, _, val = content.partition(":")
        key = key.strip().strip('"').strip("'")
        val = val.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent_indent, parent, parent_key = stack[-1]
        if val == "":
            child = {}
            if isinstance(parent, dict):
                parent[key] = child
            else:
                parent.append(child)
            stack.append((indent, child, key))
        elif val.startswith("[") and val.endswith("]"):
            inner = val[1:-1].strip()
            items = [] if not inner else [x.strip().strip('"').strip("'") for x in inner.split(",")]
            _assign(parent, key, items)
        else:
            _assign(parent, key, _scalar(val))
    return root


def _scalar(v):
    v = v.strip()
    if v in ("true", "True"):
        return True
    if v in ("false", "False"):
        return False
    if v in ("null", "None", "~"):
        return None
    if re.fullmatch(r"-?\d+", v):
        return int(v)
    if re.fullmatch(r"-?\d+\.\d+", v):
        return float(v)
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        return v[1:-1]
    return v


def _assign(container, key, value):
    if isinstance(container, dict):
        container[key] = value
    else:
        container.append(value)
