"""The minimal orchestration loop (DESIGN §3) — one ReAct-light iteration per step.

step():
  1. read the ledger,
  2. Reasoner proposes the next experiment (reason → act under search-space constraints),
  3. ENFORCE admissibility — a proposal in a ruled-out region or already tried is skipped,
     never run (this is the search-space limit binding),
  4. run it under the fidelity budget,
  5. the runner records the measured result + the prediction's calibration_error (Reflect).

This is the first point where the system actually researches by itself: it invents a
literature-grounded config, edits, validates, and remembers — with no human in the loop. It
runs end-to-end against MockLLM (no API, no GPU), so the whole cycle is unit-testable.
"""

from __future__ import annotations

import sys
from pathlib import Path

from scholarloop.advisor import Advisor
from scholarloop.agent import AgentTrace
from scholarloop.debate import DebatePanel
from scholarloop.director import Director
from scholarloop.ledger import Ledger, LedgerEntry
from scholarloop.litscout import LitScout
from scholarloop.llm import LLMClient
from scholarloop.profile import Profile
from scholarloop.reasoner import Reasoner
from scholarloop.reasoning import confidence_bound
from scholarloop.reflector import Reflector
from scholarloop.skills import SkillLibrary


class Orchestrator:
    def __init__(self, llm: LLMClient, profile: Profile, *,
                 lit_scout: LitScout | None = None, topic: str | None = None,
                 debate_panel: DebatePanel | None = None,
                 reflector: Reflector | None = None,
                 skill_library: SkillLibrary | None = None,
                 advisor: Advisor | None = None,
                 director: Director | None = None,
                 max_refines: int = 2,
                 promote_z: float = 1.0,   # how many SEMs of margin a verify result needs to reach full
                 trace: AgentTrace | None = None,
                 ledger_path: str | Path = "ledger.jsonl",
                 registry_dir: str | Path = "registry"):
        self.profile = profile
        self.trace = trace or AgentTrace()       # one shared trace: every agent call is logged
        self.reasoner = Reasoner(llm, profile, trace=self.trace)
        self.lit_scout = lit_scout
        self.debate_panel = debate_panel
        self.reflector = reflector
        self.skill_library = skill_library
        self.advisor = advisor
        self.director = director
        self.max_refines = max_refines
        self.promote_z = promote_z
        # one shared trace across EVERY agent — full auditability of the loop
        for agent in (lit_scout, reflector, advisor, director):
            if agent is not None:
                agent.trace = self.trace
        if debate_panel is not None:
            for critic in debate_panel.critics:
                critic.trace = self.trace
        self.topic = topic or profile.name
        self.ledger_path = Path(ledger_path)
        self.registry_dir = Path(registry_dir)
        self._lit: tuple[str, list[str]] | None = None   # cached (lit_context, lit_priors)
        self.guidance = ""               # Director direction / Advisor rationale for the next step
        self.last_advice: dict | None = None
        self._refine_count = 0
        self._directed = False

    def _literature(self) -> tuple[str, list[str]]:
        """Fetch literature grounding once (cached). Empty when no Lit Scout is configured."""
        if self.lit_scout is None:
            return "", []
        if self._lit is None:
            self.lit_scout.trace = self.trace
            self._lit = self.lit_scout.scout(self.topic)
        return self._lit

    def _next_id(self, entries: list[LedgerEntry]) -> str:
        # derive from the max existing suffix, not the count — robust to resumed/gapped ledgers
        nums = [int(e.id.split("_", 1)[1]) for e in entries
                if e.id.startswith("exp_") and e.id.split("_", 1)[1].isdigit()]
        return f"exp_{max(nums, default=0) + 1:04d}"

    def _parent(self, entries: list[LedgerEntry]):
        """The frontier to iterate from: the current best kept entry, else the last run."""
        best = Ledger(self.ledger_path).best(self.profile.name, self.profile.metric.direction)
        if best is not None:
            return best
        return entries[-1] if entries else None

    def _propose(self, entries: list[LedgerEntry], *, lit_context: str = "", skills: str = "",
                 lit_priors: list[str] | None = None):
        """Propose → admissibility gate → debate gate. Returns (proposal, reasoning) or None."""
        import time

        if not lit_context and lit_priors is None:
            lit_context, lit_priors = self._literature()   # fall back to the Lit Scout
        if not skills and self.skill_library is not None:
            skills = self.skill_library.render(time.time())   # inject accumulated intuition
        proposal = self.reasoner.propose(entries, lit_context=lit_context, skills=skills,
                                         guidance=self.guidance, lit_priors=lit_priors)
        if not proposal.admissible:
            print(f"orchestrator: skipping inadmissible proposal {proposal.config} "
                  f"({proposal.violations})", file=sys.stderr)
            return None
        reasoning = proposal.reasoning_blob()
        if self.debate_panel is not None:                  # diverse-perspective gate before GPU
            verdict = self.debate_panel.review(proposal)
            reasoning["debate"] = verdict.to_dict()
            if verdict.decision == "reject":
                print(f"orchestrator: panel rejected {proposal.config} ({verdict.concerns})",
                      file=sys.stderr)
                return None
        return proposal, reasoning

    def _run_tier(self, proposal, reasoning, fidelity, exp_id, parent_id, parent_score, predicted_delta):
        from scholarloop.runner import run_experiment   # imported lazily (heavy deps)
        return run_experiment(
            self.profile, fidelity, exp_id, proposal.hypothesis,
            config_override=proposal.config, edits=proposal.edits, reasoning=reasoning,
            predicted_delta=predicted_delta, parent=parent_id, parent_score=parent_score,
            ledger_path=self.ledger_path, registry_dir=self.registry_dir)

    def _post_run(self, entry, frontier, entries):
        if self.reflector is not None and self.skill_library is not None:   # Reflect → skill library
            skill = self.reflector.reflect(entry, frontier)
            if skill is not None:
                self.skill_library.add(skill)
        if self.advisor is not None:                       # post-run loop control
            recent = [e.verdict for e in entries[-5:]]
            self.last_advice = self.advisor.advise(entry, frontier, recent_verdicts=recent)
            self._apply_advice(entries)

    def step(self, *, fidelity: str = "smoke", lit_context: str = "", skills: str = "",
             lit_priors: list[str] | None = None) -> LedgerEntry | None:
        """Run one autonomous iteration at a single fidelity. Returns the entry, or None if skipped."""
        entries = list(Ledger(self.ledger_path).read_all())
        prepared = self._propose(entries, lit_context=lit_context, skills=skills, lit_priors=lit_priors)
        if prepared is None:
            return None
        proposal, reasoning = prepared
        parent = self._parent(entries)
        entry = self._run_tier(proposal, reasoning, fidelity, self._next_id(entries),
                               parent.id if parent else None,
                               parent.primary_score() if parent else None,
                               proposal.predicted_delta)
        self._post_run(entry, parent, entries)
        return entry

    def funnel_step(self, *, tiers=("smoke", "verify", "full"), lit_context: str = "",
                    skills: str = "", lit_priors: list[str] | None = None) -> list[LedgerEntry]:
        """Funnel one idea up the fidelity ladder (DESIGN §3): each tier must clear a promotion
        gate to advance, so cheap smoke runs kill most ideas before any expensive full run.
        The tiers chain via `parent`; calibration is measured against the frontier. Returns the
        chain of entries produced (1-3)."""
        entries = list(Ledger(self.ledger_path).read_all())
        prepared = self._propose(entries, lit_context=lit_context, skills=skills, lit_priors=lit_priors)
        if prepared is None:
            return []
        proposal, reasoning = prepared
        frontier = self._parent(entries)
        frontier_score = frontier.primary_score() if frontier else None
        gate = self._gate_score(frontier_score)        # must beat the frontier (or baseline) to climb
        produced: list[LedgerEntry] = []
        for i, fidelity in enumerate(tiers):
            if i == 0:                                  # smoke: child of the frontier, calibrated vs it
                parent_id = frontier.id if frontier else None
                parent_score, pred = frontier_score, proposal.predicted_delta
            else:                                       # higher tiers confirm: chain, and no new prediction
                prev = produced[-1]
                parent_id, parent_score, pred = prev.id, prev.primary_score(), None
            entry = self._run_tier(proposal, reasoning, fidelity,
                                   self._next_id(entries + produced), parent_id, parent_score, pred)
            produced.append(entry)
            if not self._promote(entry, fidelity, gate):
                break
        if produced:
            self._post_run(produced[-1], frontier, entries + produced)
        return produced

    def _gate_score(self, frontier_score: float | None) -> float | None:
        """The promotion threshold: must beat the current best (frontier), or the baseline if
        no frontier yet — whichever is more stringent. None means no gate (climb freely)."""
        baseline = self.profile.best_baseline()
        if frontier_score is None:
            return baseline
        if baseline is None:
            return frontier_score
        return frontier_score if self.profile.metric.is_better(frontier_score, baseline) else baseline

    def _promote(self, entry: LedgerEntry, fidelity: str, gate: float | None) -> bool:
        """Promotion gate (deterministic): does this tier earn a more expensive confirmation?"""
        if fidelity == "full":
            return False                                   # full is terminal
        score = entry.primary_score()
        if score is None:                                  # killed
            return False
        if gate is None:
            return True                                    # nothing to gate on yet — climb
        if not self.profile.metric.is_better(score, gate):
            return False                                   # didn't clear the bar — drop it cheaply
        if fidelity == "verify":                           # statistical-significance gate (not just the mean)
            seeds = entry.metric.get("seeds") or [score]
            bound = confidence_bound(seeds, self.profile.metric.direction, self.promote_z)
            if not self.profile.metric.is_better(bound, gate):
                return False                               # improvement isn't robust to seed noise
        return True

    def _apply_advice(self, entries: list[LedgerEntry]) -> None:
        decision = (self.last_advice or {}).get("decision", "proceed")
        rationale = (self.last_advice or {}).get("rationale", "")
        if decision == "refine":
            self._refine_count += 1
            self.guidance = rationale
            if self._refine_count >= self.max_refines:     # bounded: stop refining a dead idea
                self._pivot(entries, rationale)            # carry the real rationale into the pivot
        elif decision == "pivot":
            self._pivot(entries, rationale)
        else:  # "proceed", or any unrecognized decision -> safe default (never an accidental pivot)
            self._refine_count, self.guidance = 0, ""

    def _pivot(self, entries: list[LedgerEntry], rationale: str) -> None:
        """Change direction: reset refines, drop cached literature, re-ground.

        With a Director this genuinely re-directs (new topic + direction). Without one the topic
        is unchanged, so we at least instruct the Reasoner to leave the exhausted region.
        """
        self._refine_count = 0
        self._lit = None
        if self.director is not None:
            d = self.director.direct(entries)
            self.topic = d.get("topic", self.topic)
            self.guidance = d.get("direction", rationale)
        else:
            self.guidance = (f"{rationale} — change approach: explore a different architecture/"
                             "optimizer outside the current focus and ruled-out regions.").strip()

    def run(self, n_steps: int, *, fidelity: str = "smoke", funnel: bool = False,
            lit_context: str = "", skills: str = "") -> list[LedgerEntry]:
        """Run up to n_steps autonomous iterations; returns the entries that actually ran.

        With funnel=True each iteration funnels one idea through smoke→verify→full (cheap screen
        first); otherwise each iteration is a single run at `fidelity`.
        """
        if self.director is not None and not self._directed:   # set the campaign direction once
            d = self.director.direct(list(Ledger(self.ledger_path).read_all()))
            self.topic, self.guidance, self._directed = d.get("topic", self.topic), \
                d.get("direction", ""), True
        out: list[LedgerEntry] = []
        for _ in range(n_steps):
            if funnel:
                out.extend(self.funnel_step(lit_context=lit_context, skills=skills))
            else:
                entry = self.step(fidelity=fidelity, lit_context=lit_context, skills=skills)
                if entry is not None:
                    out.append(entry)
        return out
