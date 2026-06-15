"""Skill Library: content-hash dedup, time-decay weighting, ranked render."""

from scholarloop.skills import Skill, SkillLibrary

DAY = 86400.0


def test_make_dedup_id_is_content_addressed():
    a = Skill.make("optimizer", 0.8, "use cosine schedule", "exp_1", 0.0)
    b = Skill.make("Optimizer", 0.5, "Use Cosine Schedule", "exp_2", 100.0)  # same content, diff case
    assert a.id == b.id


def test_weight_halves_each_half_life():
    s = Skill.make("c", 0.8, "m", "exp", 0.0)
    assert s.weight(0.0) == 0.8
    assert abs(s.weight(30 * DAY) - 0.4) < 1e-9
    assert abs(s.weight(60 * DAY) - 0.2) < 1e-9


def test_library_dedup_overwrites_same_lesson(tmp_path):
    lib = SkillLibrary(tmp_path)
    lib.add(Skill.make("optimizer", 0.8, "cosine", "e1", 0.0))
    lib.add(Skill.make("optimizer", 0.9, "cosine", "e2", 100.0))   # same content -> one file
    assert len(lib.all()) == 1
    lib.add(Skill.make("arch", 0.5, "go deeper", "e3", 0.0))
    assert len(lib.all()) == 2


def test_active_ranks_by_decayed_weight(tmp_path):
    lib = SkillLibrary(tmp_path)
    lib.add(Skill.make("a", 0.9, "recent strong", "e1", 10 * DAY))
    lib.add(Skill.make("b", 0.9, "old strong", "e2", 0.0))         # same severity, older -> lower weight
    rows = lib.active(now=10 * DAY, top_k=1)
    assert rows[0][0].mitigation == "recent strong"
    assert "recent strong" in lib.render(now=10 * DAY)
