"""红外相机照片的出没事件归组与训练/验证稳定划分（纯函数，无 I/O）。

规则
----
1. 同一相机（camera_no）的照片按 (拍摄时间, 照片号) 升序排列；
   相邻两张照片间隔 <= gap_seconds 秒即归入同一"出没事件组"，
   分组具有传递性（链式桥接的照片同属一组）。
2. 组的稳定标识 group_id = 组内最早照片的 photo_no
   （时间相同取 photo_no 较小者）。
3. 训练/验证按组整体划分（同一组不会跨训练与验证，避免同一出没
   事件泄漏到两侧）。划分结果由 sha256(salt | group_id) 决定，
   对相同输入完全确定，与照片入库先后、批次顺序无关。

稳定性说明：补传照片使两组合并时，新组沿用"更早那组"的 group_id，
其划分结果保持不变；若补传照片比组内最早照片还早，group_id 会改
变为该照片号，划分可能随之翻转——这是确定性规则下的预期行为。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

DEFAULT_GAP_SECONDS = 60


@dataclass(frozen=True)
class Photo:
    photo_no: str
    camera_no: str
    captured_at: datetime  # tz-aware, UTC
    species: str
    rel_path: str


@dataclass(frozen=True)
class Group:
    group_id: str  # 组内最早照片的 photo_no
    camera_no: str
    started_at: datetime
    ended_at: datetime
    photos: tuple[Photo, ...]  # 按 (captured_at, photo_no) 升序


def compute_groups(photos: Sequence[Photo], gap_seconds: int = DEFAULT_GAP_SECONDS) -> list[Group]:
    """把照片按相机分组、按时间链式归并，返回稳定的组列表。"""
    by_camera: dict[str, list[Photo]] = {}
    for p in photos:
        by_camera.setdefault(p.camera_no, []).append(p)

    groups: list[Group] = []
    for camera_no in sorted(by_camera):
        ordered = sorted(by_camera[camera_no], key=lambda p: (p.captured_at, p.photo_no))
        current: list[Photo] = []
        prev: Photo | None = None
        for p in ordered:
            if prev is not None and (p.captured_at - prev.captured_at).total_seconds() > gap_seconds:
                groups.append(_make_group(current))
                current = []
            current.append(p)
            prev = p
        if current:
            groups.append(_make_group(current))

    groups.sort(key=lambda g: (g.started_at, g.camera_no, g.group_id))
    return groups


def _make_group(photos: list[Photo]) -> Group:
    first = photos[0]  # 调用前已按 (captured_at, photo_no) 排序
    return Group(
        group_id=first.photo_no,
        camera_no=first.camera_no,
        started_at=first.captured_at,
        ended_at=photos[-1].captured_at,
        photos=tuple(photos),
    )


def assign_split(group_id: str, val_percent: int, salt: str) -> str:
    """确定性划分：返回 "train" 或 "val"。同一 group_id 永远得到同一结果。"""
    digest = hashlib.sha256(f"{salt}|{group_id}".encode("utf-8")).hexdigest()
    bucket = int(digest, 16) % 100
    return "val" if bucket < val_percent else "train"
