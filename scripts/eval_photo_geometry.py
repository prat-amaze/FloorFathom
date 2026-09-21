"""Score a finished photo-tier plan.json against tape ground truth: wall length, corner position,
floor area, ceiling height error."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from floorfathom.schema import CapturePlan


def score_geometry(plan: CapturePlan, ground_truth: dict) -> dict:
    room = plan.rooms[0]
    gt = ground_truth.get(plan.capture, {})
    wall_err = []
    for w in room.walls:
        truth = gt.get("walls", {}).get(w.id)
        if truth is not None and w.length.value is not None:
            wall_err.append(abs(w.length.value - truth) / truth)
    corner_err = []
    gt_corners = gt.get("corners")
    if gt_corners:
        import numpy as np
        for p, t in zip(room.polygon, gt_corners):
            corner_err.append(float(np.hypot(p[0] - t[0], p[1] - t[1])))
    area_err = None
    if room.floor_area.value is not None and gt.get("floor_area") is not None:
        area_err = abs(room.floor_area.value - gt["floor_area"]) / gt["floor_area"]
    ceiling_err = None
    if room.ceiling_height.value is not None and gt.get("ceiling_height") is not None:
        ceiling_err = abs(room.ceiling_height.value - gt["ceiling_height"])
    return {"wall_length_error": wall_err, "corner_position_error": corner_err,
            "floor_area_error": area_err, "ceiling_height_error": ceiling_err}


def main(plan_path: str, ground_truth_path: str) -> None:
    plan = CapturePlan.model_validate_json(Path(plan_path).read_text())
    truth = json.loads(Path(ground_truth_path).read_text())
    for room in plan.rooms:
        scores = score_geometry(plan.model_copy(update={"rooms": [room]}), truth)
        print(f"{room.name}: wall errors {[f'{e:.1%}' for e in scores['wall_length_error']]}, "
              f"area error {scores['floor_area_error']}, ceiling error {scores['ceiling_height_error']}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
