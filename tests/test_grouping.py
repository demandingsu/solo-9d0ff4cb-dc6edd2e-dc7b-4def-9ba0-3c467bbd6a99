"""归组与稳定划分的纯逻辑测试。"""
from datetime import datetime, timedelta, timezone

from app.grouping import Photo, assign_split, compute_groups

T0 = datetime(2026, 3, 1, 8, 0, 0, tzinfo=timezone.utc)


def p(no, cam, sec, species="boar"):
    return Photo(no, cam, T0 + timedelta(seconds=sec), species, f"{cam}/{no}.jpg")


def test_transitive_chaining_merges_via_bridge():
    # A..B 间隔 50s，B..C 间隔 50s；A..C 相隔 100s 仍同组（传递归组）
    photos = [p("P1", "C1", 0), p("P2", "C1", 50), p("P3", "C1", 100)]
    groups = compute_groups(photos)
    assert len(groups) == 1
    assert [x.photo_no for x in groups[0].photos] == ["P1", "P2", "P3"]


def test_gap_over_60s_splits():
    photos = [p("P1", "C1", 0), p("P2", "C1", 61)]
    assert len(compute_groups(photos)) == 2


def test_exactly_60s_stays_together():
    photos = [p("P1", "C1", 0), p("P2", "C1", 60)]
    assert len(compute_groups(photos)) == 1


def test_cameras_are_independent():
    photos = [p("P1", "C1", 0), p("P2", "C2", 10)]
    groups = compute_groups(photos)
    assert len(groups) == 2
    assert {g.camera_no for g in groups} == {"C1", "C2"}


def test_group_id_is_earliest_photo():
    photos = [p("P9", "C1", 30), p("P3", "C1", 0)]
    (g,) = compute_groups(photos)
    assert g.group_id == "P3"


def test_backfill_bridge_merges_and_keeps_earlier_group_id():
    base = [p("P1", "C1", 0), p("P2", "C1", 30), p("P3", "C1", 100), p("P4", "C1", 130)]
    assert len(compute_groups(base)) == 2  # 30s -> 100s 间隔 70s，分为两组
    merged = compute_groups(base + [p("P5", "C1", 65)])  # 桥接：30->65=35s, 65->100=35s
    assert len(merged) == 1
    assert merged[0].group_id == "P1"  # 沿用更早组的标识


def test_split_is_deterministic_and_group_consistent():
    photos = [p(f"P{i}", "C1", i * 300) for i in range(1, 21)]  # 20 个独立组
    groups1 = compute_groups(photos)
    groups2 = compute_groups(list(reversed(photos)))
    s1 = [assign_split(g.group_id, 20, "salt") for g in groups1]
    s2 = [assign_split(g.group_id, 20, "salt") for g in groups2]
    assert s1 == s2  # 与输入顺序无关
    assert "val" in s1 and "train" in s1  # 20 个组应两侧都有
