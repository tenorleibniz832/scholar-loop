"""Skill Library (DESIGN §2.4 / §4.5 mechanism 3) — the PhD's accumulating intuition.

A deterministic store of lessons distilled from past runs. Each lesson decays in influence
over time (half-life ~30 days), so recent experience outweighs stale experience without ever
being deleted. The Reflector writes lessons here; the Reasoner reads the top-weighted ones
into its prompt next round. No training, backbone-agnostic — just prompt overlays.

Dedup is by content hash: the same lesson (category + mitigation) maps to the same file, so a
judge-rejected direction can't silently reappear as a fresh skill every round — it refreshes
in place. The store is plain files under a directory (`arc-<id>.json`), trivially inspectable.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Skill:
    id: str
    category: str
    severity: float       # (0, 1] — how strongly this lesson should weigh in
    mitigation: str       # the actionable lesson, written for the next reasoning prompt
    source: str           # the experiment id that produced it
    ts: float             # when the originating experiment ran (decay is measured from here)

    @staticmethod
    def make(category: str, severity: float, mitigation: str, source: str, ts: float) -> "Skill":
        key = f"{category.strip().lower()}|{mitigation.strip().lower()}"
        sid = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
        return Skill(id=sid, category=category, severity=float(severity),
                     mitigation=mitigation, source=source, ts=ts)

    def weight(self, now: float, half_life_days: float = 30.0) -> float:
        """Time-decayed influence: severity halves every `half_life_days`."""
        dt_days = max(0.0, (now - self.ts) / 86400.0)
        return self.severity * (2.0 ** (-dt_days / half_life_days))


class SkillLibrary:
    def __init__(self, path: str | Path, *, half_life_days: float = 30.0):
        self.dir = Path(path)
        self.half_life_days = half_life_days

    def add(self, skill: Skill) -> Skill:
        """Write (or refresh in place, by content id) a lesson."""
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / f"arc-{skill.id}.json").write_text(
            json.dumps(asdict(skill), indent=2, ensure_ascii=False))
        return skill

    def all(self) -> list[Skill]:
        if not self.dir.exists():
            return []
        out = []
        for f in sorted(self.dir.glob("arc-*.json")):
            try:
                out.append(Skill(**json.loads(f.read_text())))
            except (json.JSONDecodeError, TypeError):
                continue  # skip a corrupt skill file rather than break the read
        return out

    def active(self, now: float | None = None, *, top_k: int | None = None,
               min_weight: float = 0.0) -> list[tuple[Skill, float]]:
        """(skill, weight) pairs sorted by decayed weight, filtered/capped."""
        now = time.time() if now is None else now
        scored = [(s, s.weight(now, self.half_life_days)) for s in self.all()]
        scored = [(s, w) for s, w in scored if w > min_weight]
        scored.sort(key=lambda sw: sw[1], reverse=True)
        return scored[:top_k] if top_k is not None else scored

    def render(self, now: float | None = None, *, top_k: int = 5) -> str:
        """Render the top lessons as a prompt block for the Reasoner's `skills` slot."""
        rows = self.active(now, top_k=top_k)
        if not rows:
            return ""
        return "\n".join(f"- [{s.category}, w={w:.2f}] {s.mitigation}" for s, w in rows)
