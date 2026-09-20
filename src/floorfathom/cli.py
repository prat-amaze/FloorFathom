"""Command line: ``floorfathom plan <capture_dir> --out <dir>``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .models import REGISTRY, ModelError, fetch
from .pipeline import run, schema_json


def _summary(plan) -> str:
    lines = [f"{plan.capture} ({plan.tier}): {len(plan.rooms)} room(s), {plan.diagnostics.seconds:.1f} s"]
    for r in plan.rooms:
        a, c = r.floor_area, r.ceiling_height
        ch = "n/a" if c.value is None else f"{c.value:.3f} [{c.lo:.3f}, {c.hi:.3f}] m"
        flags = f"  flags: {', '.join(r.flags)}" if r.flags else ""
        lines.append(f"  {r.id}: area {a.value:.2f} [{a.lo:.2f}, {a.hi:.2f}] m2, ceiling {ch}, {len(r.walls)} walls, {len(r.openings)} openings{flags}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="floorfathom", description="Handheld capture to dimensioned floor plan.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="run the pipeline on one capture folder")
    p.add_argument("capture", type=Path, help="capture folder")
    p.add_argument("--out", type=Path, default=Path("out"), help="output folder (default: out)")
    p.add_argument("--tier", choices=["lidar", "video", "photo"], help="input tier (default: detect from the folder)")
    p.add_argument("--bootstrap", type=int, default=20, help="bootstrap replicates for the intervals (default: 20)")
    p.add_argument("--seed", type=int, default=0, help="random seed for the bootstrap (default: 0)")
    p.add_argument("--no-debug", action="store_true", help="skip the debug folder (point cloud, top-down view)")

    sub.add_parser("schema", help="print the JSON Schema of the output")

    f = sub.add_parser("fetch-models", help="download the pretrained models once (the only step that needs the network)")
    f.add_argument("--model", choices=sorted(REGISTRY), help="fetch one model (default: all)")

    args = ap.parse_args(argv)
    if args.cmd == "schema":
        print(schema_json())
        return 0
    if args.cmd == "fetch-models":
        try:
            for name in [args.model] if args.model else sorted(REGISTRY):
                spec = REGISTRY[name]
                print(f"{name}: {spec.repo_id} @ {spec.revision[:10]}, licence: {spec.licence}")
                print(f"  ready in {fetch(name)}")
        except ModelError as e:
            print(f"error: {e}", file=sys.stderr)
            return 3
        return 0
    if not args.capture.is_dir():
        print(f"error: {args.capture} is not a folder", file=sys.stderr)
        return 2
    try:
        plan = run(args.capture, args.out, tier=args.tier, replicates=args.bootstrap, seed=args.seed, debug=not args.no_debug)
    except NotImplementedError as e:
        print(f"error: {e}", file=sys.stderr)
        return 3
    print(_summary(plan))
    print(f"wrote {args.out / 'plan.json'} and {args.out / 'plan.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
