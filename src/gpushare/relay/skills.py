from __future__ import annotations

import re
from pathlib import Path

REQUIRED_SKILLS = (
    "training-optimizer",
    "inference-optimizer",
    "chip-migration",
    "job-allocation",
    "quality-verifier",
    "provider-onboarding",
)


class SkillRegistry:
    def __init__(self, root: Path):
        self.root = root

    def load(self) -> list[dict[str, str]]:
        loaded = []
        for name in REQUIRED_SKILLS:
            path = self.root / name / "SKILL.md"
            if not path.is_file():
                raise FileNotFoundError(f"required Relay skill is missing: {path}")
            text = path.read_text(encoding="utf-8")
            match = re.match(r"---\s*\nname:\s*([^\n]+)\ndescription:\s*([^\n]+)\n---", text)
            if not match or match.group(1).strip() != name:
                raise ValueError(f"invalid skill frontmatter: {path}")
            loaded.append(
                {
                    "name": name,
                    "description": match.group(2).strip(),
                    "path": str(path.relative_to(self.root.parent)),
                }
            )
        return loaded
