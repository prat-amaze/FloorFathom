import json
from pathlib import Path

from floorfathom.schema import json_schema

ROOT = Path(__file__).resolve().parent.parent


def test_published_schema_matches_the_models():
    """schema/capture_plan.schema.json is the published contract; regenerate it with
    `uv run floorfathom schema > schema/capture_plan.schema.json` when the models change."""
    committed = json.loads((ROOT / "schema" / "capture_plan.schema.json").read_text())
    assert committed == json_schema()
