"""Parse an Isaac Sim camera property export into a Python-friendly file.

Standalone (stdlib only):
  python parse_camera_config.py                       # -> configs/cameras.py
  python parse_camera_config.py --format json         # -> configs/cameras.json
  python parse_camera_config.py -i props.txt -d defaults.json -o out.py --strict

Also importable: parse_camera_config.generate(...) / build_cameras(...).

Missing properties are taken from the defaults file (a warning is printed for each).
Camera orientation is the exported quaternion xformOp:orient (w, x, y, z); the camera looks
along local -Z with +Y up, and "rotation" is the camera-to-world 3x3 matrix.
"""
import argparse, ast, json, math, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "configs" / "camera_properties.txt"
DEFAULT_DEFAULTS = HERE / "configs" / "camera_defaults.json"
DEFAULT_OUTPUT = HERE / "configs" / "cameras.py"

HEADER_RE = re.compile(r"^===\s*CAMERA:\s*(\S+)\s*===\s*$")
# USD property name -> output key
FIELDS = {
    "xformOp:translate": "position",
    "xformOp:orient": "quat_wxyz",
    "xformOp:scale": "scale",
    "focalLength": "focal_length",
    "horizontalAperture": "horizontal_aperture",
    "verticalAperture": "vertical_aperture",
    "resolution": "resolution",
    "clippingRange": "clipping_range",
    "focusDistance": "focus_distance",
    "projection": "projection",
    "exposure:fStop": "f_stop",
    "exposure:responsivity": "responsivity",
    "exposure:time": "exposure_time",
    "omni:kit:cameraLock": "camera_lock",
    "omni:kit:centerOfInterest": "center_of_interest",
}
EXPECTED_XFORM_ORDER = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]


def _value(text):
    """'(1, 2)' -> tuple, '5.0' -> float, 'True' -> bool, 'perspective' -> str, '[a, b]' -> list[str]."""
    text = text.strip()
    try:
        v = ast.literal_eval(text)
        return tuple(v) if isinstance(v, list) else v
    except (ValueError, SyntaxError):
        if text.startswith("[") and text.endswith("]"):
            return [s.strip() for s in text[1:-1].split(",") if s.strip()]
        return text


def parse_properties(text):
    """Return {camera_path: {usd_property: value}} in file order."""
    cams, cur = {}, None
    for n, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        m = HEADER_RE.match(line)
        if m:
            cur = cams.setdefault(m.group(1), {})
        elif cur is None:
            raise ValueError(f"line {n}: property before any '=== CAMERA: ... ===' header")
        elif ": " in line:
            k, v = line.split(": ", 1)  # USD names contain ':' but never ': '
            cur[k.strip()] = _value(v)
        else:
            raise ValueError(f"line {n}: cannot parse {line!r}")
    return cams


def _norm(q):
    n = math.sqrt(sum(c * c for c in q))
    if n < 1e-12:
        raise ValueError(f"zero-length quaternion {q}")
    return tuple(c / n for c in q)


def quat_to_matrix(q):
    """Unit quaternion (w, x, y, z) -> 3x3 rotation matrix (row tuples)."""
    w, x, y, z = _norm(q)
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


def _cam_id(path):
    m = re.search(r"(\d+)$", path.rstrip("/").split("/")[-1])
    if not m:
        raise ValueError(f"cannot get camera id from {path!r} (name must end with digits)")
    return int(m.group(1))


def build_cameras(raw, defaults, strict=False, warn=print):
    """Merge parsed properties with defaults and derive pose + intrinsics. Returns {id: dict}."""
    cams, used = {}, {}  # used: property -> camera names that fell back to the default
    for path, props in raw.items():
        cid, name = _cam_id(path), path.rstrip("/").split("/")[-1]
        if cid in cams:
            raise ValueError(f"duplicate camera id {cid} ({path})")
        merged = {}
        for usd, key in FIELDS.items():
            if usd in props:
                merged[key] = props[usd]
            elif usd in defaults:
                if strict:
                    raise ValueError(f"{name}: property '{usd}' not found (--strict)")
                used.setdefault(usd, []).append(name)
                v = defaults[usd]
                merged[key] = tuple(v) if isinstance(v, list) else v
            else:
                raise ValueError(f"{name}: property '{usd}' not found and no default available")
        order = props.get("xformOpOrder")
        if order is not None and list(order) != EXPECTED_XFORM_ORDER:
            warn(f"warning: {name}: xformOpOrder {list(order)} != {EXPECTED_XFORM_ORDER}; "
                 "assuming translate/orient/scale anyway")
        if any(abs(s - 1) > 1e-9 for s in merged["scale"]):
            warn(f"warning: {name}: non-unit scale {merged['scale']} ignored")
        q = _norm(merged["quat_wxyz"])
        R = quat_to_matrix(q)
        W, H = merged["resolution"]
        hfov = 2 * math.degrees(math.atan(merged["horizontal_aperture"] / 2 / merged["focal_length"]))
        cams[cid] = {
            "id": cid,
            "name": name,
            "path": path,
            "position": tuple(float(c) for c in merged["position"]),
            "quat_wxyz": q,
            "rotation": R,                                   # camera -> world
            "forward": tuple(-R[i][2] for i in range(3)),    # R @ [0, 0, -1]
            "up": tuple(R[i][1] for i in range(3)),          # R @ [0, 1, 0]
            "resolution": (int(W), int(H)),
            "focal_length": float(merged["focal_length"]),
            "horizontal_aperture": float(merged["horizontal_aperture"]),
            "vertical_aperture": float(merged["vertical_aperture"]),
            "hfov_deg": hfov,
            "focal_px": W * merged["focal_length"] / merged["horizontal_aperture"],
            "clipping_range": tuple(merged["clipping_range"]),
            "focus_distance": float(merged["focus_distance"]),
            "projection": merged["projection"],
            "f_stop": float(merged["f_stop"]),
            "responsivity": float(merged["responsivity"]),
            "exposure_time": float(merged["exposure_time"]),
            "camera_lock": bool(merged["camera_lock"]),
            "center_of_interest": tuple(merged["center_of_interest"]),
        }
    for usd, names in used.items():
        who = "all cameras" if len(names) == len(raw) else ", ".join(names)
        warn(f"warning: '{usd}' not found for {who} -> default {defaults[usd]!r}")
    if not cams:
        raise ValueError("no cameras found")
    ids = sorted(cams)
    missing = sorted(set(range(ids[0], ids[-1] + 1)) - set(ids))
    if missing:
        warn(f"warning: camera ids not contiguous, missing {missing}")
    return {i: cams[i] for i in ids}


def _fmt(v, ind=0):
    if isinstance(v, dict):
        pad = " " * (ind + 4)
        return "{\n" + "".join(f"{pad}{k!r}: {_fmt(x, ind + 4)},\n" for k, x in v.items()) + " " * ind + "}"
    if isinstance(v, tuple) and v and isinstance(v[0], tuple):
        pad = " " * (ind + 4)
        return "(\n" + "".join(f"{pad}{_fmt(x)},\n" for x in v) + " " * ind + ")"
    if isinstance(v, tuple):
        return "(" + ", ".join(_fmt(x) for x in v) + ("," if len(v) == 1 else "") + ")"
    return repr(v)


def write_py(cams, out, src):
    body = (f'"""AUTO-GENERATED by parse_camera_config.py from {src}. Do not edit.\n\n'
            "CAMERAS[id]: position (m), quat_wxyz, rotation (camera->world 3x3), forward, up,\n"
            "resolution (px), focal_length / apertures (mm), hfov_deg, focal_px, ...\n"
            '"""\n\n'
            f"CAMERAS = {_fmt(cams)}\n")
    Path(out).write_text(body)


def write_json(cams, out, src):
    def conv(v):
        return [conv(x) for x in v] if isinstance(v, tuple) else v
    data = {"_generated_by": f"parse_camera_config.py from {src}",
            "cameras": {str(i): {k: conv(v) for k, v in c.items()} for i, c in cams.items()}}
    Path(out).write_text(json.dumps(data, indent=2) + "\n")


def generate(inp=DEFAULT_INPUT, defaults=DEFAULT_DEFAULTS, out=DEFAULT_OUTPUT, fmt="py",
             strict=False, warn=print):
    """Parse `inp`, apply `defaults`, write `out` (py or json). Returns the cameras dict."""
    inp, out = Path(inp), Path(out)
    with open(defaults) as f:
        dflt = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    cams = build_cameras(parse_properties(inp.read_text()), dflt, strict, warn)
    out.parent.mkdir(parents=True, exist_ok=True)
    (write_py if fmt == "py" else write_json)(cams, out, inp.name)
    return cams


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--input", default=str(DEFAULT_INPUT), help="Isaac Sim camera property export")
    p.add_argument("-d", "--defaults", default=str(DEFAULT_DEFAULTS), help="defaults JSON")
    p.add_argument("-o", "--output", help="output file (default: configs/cameras.<format>)")
    p.add_argument("--format", choices=["py", "json"], default="py")
    p.add_argument("--strict", action="store_true", help="error instead of using a default")
    p.add_argument("--quiet", action="store_true", help="no warnings / summary")
    a = p.parse_args(argv)
    out = a.output or str(DEFAULT_OUTPUT.with_suffix("." + a.format))
    warn = (lambda *_: None) if a.quiet else (lambda m: print(m, file=sys.stderr))
    try:
        cams = generate(a.input, a.defaults, out, a.format, a.strict, warn)
    except (OSError, ValueError) as e:
        sys.exit(f"error: {e}")
    if not a.quiet:
        print(f"parsed {len(cams)} cameras ({min(cams)}..{max(cams)}) -> {out}")


if __name__ == "__main__":
    main()
