"""版本索引的校验、可信恢复依据与恢复。

设计要点
--------
* 发布新版本时，两份索引的字节会被复制到 data/recovery/v{N}/ 作为独立
  可信副本，并把 sha256 + 大小记入 index_baselines（与版本发布同一事务）。
* 校验永远以"已记录的可信摘要"为准，绝不根据当前照片表重算索引。
* 任一索引缺失/被改动 -> 该版本两份下载一起暂停（状态按磁盘实时计算，
  不依赖可能过期的数据库字段）。
* 恢复只接受与可信摘要逐字节一致的内容：上传原文件，或从可信副本复制。
  恢复是逐文件原子替换；只有两份都校验通过后才恢复下载，因此不会暴露
  "半恢复"的索引。中断后重试可继续完成。
* 旧版（本功能上线前发布）：两份文件都在时可显式"按现存内容建档"；
  任一文件缺失时只提示无法恢复，不重算。
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import shutil
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

INDEX_KINDS = ("train", "val")
INDEX_FILENAMES = {"train": "train_index.csv", "val": "val_index.csv"}
INDEX_LABELS = {"train": "训练索引", "val": "验证索引"}

STATUS_OK = "ok"
STATUS_CORRUPT = "corrupt"
STATUS_MISSING = "missing"
STATUS_UNBASELINED = "unbaselined"
STATUS_BASELINE_BROKEN = "baseline_broken"

STATUS_LABELS = {
    STATUS_OK: "校验通过",
    STATUS_CORRUPT: "内容被改动",
    STATUS_MISSING: "文件缺失",
    STATUS_UNBASELINED: "未建档",
    STATUS_BASELINE_BROKEN: "恢复依据损坏",
}


class IntegrityError(Exception):
    """用户可理解的校验/恢复失败（路由转成页面提示）。"""


def utcnow_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------- 摘要与原子写

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write(path: Path, data: bytes) -> None:
    """同目录临时文件 + fsync + os.replace：读者只会看到旧字节或完整新字节。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".tmp-{path.name}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def file_digest(path: Path | None) -> tuple[str | None, int | None, str | None]:
    """返回 (sha256, size, error)。不存在返回 (None, None, None)。"""
    if path is None:
        return None, None, "路径为空"
    try:
        if not path.is_file():
            return None, None, None
        h = hashlib.sha256()
        size = 0
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
                size += len(chunk)
        return h.hexdigest(), size, None
    except OSError as exc:
        return None, None, f"无法读取: {exc}"


@contextmanager
def exclusive_lock(settings) -> Iterator[None]:
    """跨线程/跨进程串行化建档与恢复（发布不修改旧版本文件，无需此锁）。"""
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    with open(settings.lock_path, "ab") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


# ---------------------------------------------------------------- 数据访问

def get_version(conn: sqlite3.Connection, vid: int):
    return conn.execute("SELECT * FROM versions WHERE id=?", (vid,)).fetchone()


def get_baseline(conn: sqlite3.Connection, vid: int, kind: str):
    return conn.execute(
        "SELECT * FROM index_baselines WHERE version_id=? AND kind=?", (vid, kind)
    ).fetchone()


def index_rel(version, kind: str) -> str:
    """版本行里的索引相对路径；旧库可能为空，回退到约定位置。"""
    rel = version["train_path"] if kind == "train" else version["val_path"]
    if rel:
        return rel
    return f"versions/v{int(version['id'])}/{INDEX_FILENAMES[kind]}"


def _event(conn, vid, action, status, source, message) -> None:
    conn.execute(
        "INSERT INTO recovery_events(version_id, created_at, action, status, source, message)"
        " VALUES (?,?,?,?,?,?)",
        (vid, utcnow_str(), action, status, source, message[:1000]),
    )


def _set_verification(conn, vid: int, integrity: "VersionIntegrity") -> None:
    conn.execute(
        "UPDATE versions SET integrity_status=?, integrity_detail=? WHERE id=?",
        (integrity.status, integrity.summary[:1000], vid),
    )
    for kind, st in integrity.states.items():
        conn.execute(
            "INSERT INTO index_checks(version_id, kind, status, sha256, checked_at, detail)"
            " VALUES (?,?,?,?,?,?)"
            " ON CONFLICT(version_id, kind) DO UPDATE SET"
            " status=excluded.status, sha256=excluded.sha256,"
            " checked_at=excluded.checked_at, detail=excluded.detail",
            (vid, kind, st.status, st.actual_sha256, utcnow_str(), st.detail),
        )


# ---------------------------------------------------------------- 实时校验

@dataclass(frozen=True)
class IndexState:
    kind: str
    label: str
    filename: str
    rel_path: str
    status: str
    actual_sha256: str | None
    actual_size: int | None
    recorded_sha256: str | None
    copy_ok: bool
    detail: str

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK

    @property
    def can_upload(self) -> bool:
        """有可信摘要才接受上传恢复（未建档版本不能随便覆盖）。"""
        return self.recorded_sha256 is not None

    @property
    def can_restore_copy(self) -> bool:
        return (
            self.recorded_sha256 is not None
            and self.copy_ok
            and self.actual_sha256 != self.recorded_sha256
        )

    @property
    def can_repair_copy(self) -> bool:
        """当前文件仍是原字节、只是可信副本损坏 -> 可从当前文件修复副本。"""
        return (
            self.recorded_sha256 is not None
            and self.actual_sha256 == self.recorded_sha256
            and not self.copy_ok
        )


@dataclass(frozen=True)
class VersionIntegrity:
    version_id: int
    status: str
    states: dict[str, IndexState] = field(default_factory=dict)
    summary: str = ""

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK

    @property
    def downloads_allowed(self) -> bool:
        """两份都可下载才放行；任一份异常 -> 一起暂停。

        未建档的旧版本：两份文件都可读时按原有行为继续提供下载（页面提示建档），
        任一文件缺失则暂停——此时无法校验、也可能无法恢复。
        """
        return all(
            st.status == STATUS_OK
            or (st.status == STATUS_UNBASELINED and st.actual_sha256 is not None)
            for st in self.states.values()
        )

    @property
    def can_adopt(self) -> bool:
        """旧版建档：两份现存文件都可读，且尚无任何可信摘要。"""
        return all(
            st.recorded_sha256 is None and st.actual_sha256 is not None
            for st in self.states.values()
        )

    @property
    def missing_kinds(self) -> tuple[str, ...]:
        return tuple(k for k, st in self.states.items() if st.status == STATUS_MISSING)


def inspect_version(settings, conn: sqlite3.Connection, version) -> VersionIntegrity:
    """按磁盘现状 + 已记录摘要实时计算。只读，不写数据库。"""
    vid = int(version["id"])
    states: dict[str, IndexState] = {}
    for kind in INDEX_KINDS:
        rel = index_rel(version, kind)
        path = settings.data_dir / rel
        actual_sha, actual_size, read_err = file_digest(path)
        baseline = get_baseline(conn, vid, kind)
        recorded = baseline["sha256"] if baseline is not None else None

        copy_ok = False
        if baseline is not None:
            copy_sha, _, _ = file_digest(settings.data_dir / baseline["rel_path"])
            copy_ok = copy_sha == recorded

        if baseline is None:
            if actual_sha is not None:
                status = STATUS_UNBASELINED
                detail = "两份索引虽可读，但尚未建立可信摘要；请先按现存内容建档"
            else:
                status = STATUS_MISSING
                detail = "索引文件缺失，且无建档，无法恢复原始字节"
        elif actual_sha is None:
            status = STATUS_MISSING
            detail = "索引文件缺失" + (f"（{read_err}）" if read_err else "")
        elif actual_sha != recorded or actual_size != baseline["size"]:
            status = STATUS_CORRUPT
            detail = "内容与发布时的可信摘要不一致"
        elif not copy_ok:
            status = STATUS_BASELINE_BROKEN
            detail = "文件本身正确，但可信恢复副本缺失或损坏"
        else:
            status = STATUS_OK
            detail = "与原发布字节一致"

        states[kind] = IndexState(
            kind=kind,
            label=INDEX_LABELS[kind],
            filename=INDEX_FILENAMES[kind],
            rel_path=rel,
            status=status,
            actual_sha256=actual_sha,
            actual_size=actual_size,
            recorded_sha256=recorded,
            copy_ok=copy_ok,
            detail=detail,
        )

    if all(st.ok for st in states.values()):
        status = STATUS_OK
        summary = "两份索引均与原发布字节一致"
    elif all(
        st.status == STATUS_UNBASELINED for st in states.values()
    ):
        status = STATUS_UNBASELINED
        summary = "旧版本尚未建档：两份索引可读，但发布时未留存可信摘要，可按现存内容建档"
    else:
        bad = [
            f"{INDEX_LABELS[k]}：{STATUS_LABELS[st.status]}"
            for k, st in states.items()
            if not st.ok
        ]
        status = "blocked"
        summary = "；".join(bad) + "（该版本两份下载均已暂停）"
    return VersionIntegrity(vid, status, states, summary)


def refresh_statuses(settings, conn: sqlite3.Connection) -> None:
    """启动时把实时校验结果写入审计字段（不产生事件，不改变文件）。"""
    rows = conn.execute("SELECT * FROM versions").fetchall()
    if not rows:
        return
    conn.execute("BEGIN IMMEDIATE")
    try:
        for v in rows:
            integrity = inspect_version(settings, conn, v)
            _set_verification(conn, int(v["id"]), integrity)
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise


def record_verification(settings, conn: sqlite3.Connection, vid: int) -> VersionIntegrity:
    version = get_version(conn, vid)
    if version is None:
        raise IntegrityError("版本不存在")
    integrity = inspect_version(settings, conn, version)
    try:
        conn.execute("BEGIN IMMEDIATE")
        _set_verification(conn, vid, integrity)
        _event(conn, vid, "verify", "ok" if integrity.ok else "failed", "disk", integrity.summary)
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    return integrity


# ---------------------------------------------------------------- 可信恢复依据

def _snapshot_dir(settings, vid: int, payload: dict[str, bytes]) -> Path:
    """把两份原字节写入 recovery/v{N}/（tmp 目录 + rename，整目录原子出现）。"""
    target = settings.recovery_dir / f"v{vid}"
    tmp = settings.recovery_dir / f".tmp-v{vid}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        for kind, data in payload.items():
            path = tmp / INDEX_FILENAMES[kind]
            with open(path, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
        _fsync_dir(tmp)
        if target.exists():
            # 只有确认库中该版本尚无任何摘要时才可能是上次中断的孤儿目录。
            shutil.rmtree(target)
        os.rename(tmp, target)
        _fsync_dir(settings.recovery_dir)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return target


def create_baseline(
    settings,
    conn: sqlite3.Connection,
    vid: int,
    train_rel: str,
    val_rel: str,
    source: str,
) -> None:
    """在当前写事务内为两份索引留存可信恢复依据。调用方负责提交/回滚。

    失败会抛异常；调用方回滚事务并清理 recovery/v{N}。
    """
    if conn.execute(
        "SELECT 1 FROM index_baselines WHERE version_id=? LIMIT 1", (vid,)
    ).fetchone():
        raise IntegrityError(f"版本 v{vid} 已有可信恢复依据")

    payload: dict[str, bytes] = {}
    rels = {"train": train_rel, "val": val_rel}
    for kind in INDEX_KINDS:
        path = settings.data_dir / rels[kind]
        try:
            payload[kind] = path.read_bytes()
        except OSError as exc:
            raise IntegrityError(f"无法读取{INDEX_LABELS[kind]}用于留存恢复依据: {exc}") from exc

    _snapshot_dir(settings, vid, payload)
    now = utcnow_str()
    recovery_dir = settings.recovery_dir / f"v{vid}"
    for kind in INDEX_KINDS:
        data = payload[kind]
        rel = recovery_dir.relative_to(settings.data_dir) / INDEX_FILENAMES[kind]
        conn.execute(
            "INSERT INTO index_baselines"
            "(version_id, kind, sha256, size, rel_path, source, created_at)"
            " VALUES (?,?,?,?,?,?,?)",
            (vid, kind, sha256_bytes(data), len(data), str(rel), source, now),
        )
    # 发布后立即记录“两份均通过”的校验状态，并留下事件。
    integrity = inspect_version(settings, conn, get_version(conn, vid))
    if not integrity.ok:  # 理论上刚写入的文件必然一致；不一致则让整个发布回滚
        raise IntegrityError(f"发布后校验失败：{integrity.summary}")
    _set_verification(conn, vid, integrity)
    _event(conn, vid, "publish", "ok", "publish", "发布时留存两份索引的可信恢复依据")


# ---------------------------------------------------------------- 旧版建档

def adopt_version(settings, conn: sqlite3.Connection, vid: int) -> str:
    """旧版：按现存两份文件建档。任一缺失则拒绝，绝不根据照片表重算。"""
    with exclusive_lock(settings):
        version = get_version(conn, vid)
        if version is None:
            raise IntegrityError("版本不存在")
        if conn.execute(
            "SELECT 1 FROM index_baselines WHERE version_id=? LIMIT 1", (vid,)
        ).fetchone():
            raise IntegrityError(f"版本 v{vid} 已建档，无需重复操作")

        rels = {k: index_rel(version, k) for k in INDEX_KINDS}
        payload: dict[str, bytes] = {}
        missing = [
            INDEX_LABELS[k] for k in INDEX_KINDS
            if not (settings.data_dir / rels[k]).is_file()
        ]
        if missing:
            msg = (
                "无法恢复：" + "、".join(missing) + "文件缺失。"
                "不会根据当前照片表重算索引；缺少原字节即无法恢复该版本。"
            )
            conn.execute("BEGIN IMMEDIATE")
            _event(conn, vid, "adopt", "rejected", "disk", msg)
            conn.execute("COMMIT")
            raise IntegrityError(msg)

        for kind in INDEX_KINDS:
            try:
                payload[kind] = (settings.data_dir / rels[kind]).read_bytes()
            except OSError as exc:
                msg = f"无法读取{INDEX_LABELS[kind]}：{exc}"
                conn.execute("BEGIN IMMEDIATE")
                _event(conn, vid, "adopt", "rejected", "disk", msg)
                conn.execute("COMMIT")
                raise IntegrityError(msg) from exc

        conn.execute("BEGIN IMMEDIATE")
        try:
            if conn.execute(
                "SELECT 1 FROM index_baselines WHERE version_id=? LIMIT 1", (vid,)
            ).fetchone():
                raise IntegrityError(f"版本 v{vid} 已建档（并发操作），未重复写入")
            recovery_dir = settings.recovery_dir / f"v{vid}"
            _snapshot_dir(settings, vid, payload)
            now = utcnow_str()
            for kind in INDEX_KINDS:
                data = payload[kind]
                rel = recovery_dir.relative_to(settings.data_dir) / INDEX_FILENAMES[kind]
                conn.execute(
                    "INSERT INTO index_baselines"
                    "(version_id, kind, sha256, size, rel_path, source, created_at)"
                    " VALUES (?,?,?,?,?,?,?)",
                    (vid, kind, sha256_bytes(data), len(data), str(rel), "adopt", now),
                )
            integrity = inspect_version(settings, conn, get_version(conn, vid))
            _set_verification(conn, vid, integrity)
            _event(
                conn, vid, "adopt", "ok", "disk",
                "已按现存两份索引建档；此后任何改动都会被检出。",
            )
            conn.execute("COMMIT")
            return f"版本 v{vid} 已按现存内容建档（可信摘要已记录）"
        except BaseException:
            conn.execute("ROLLBACK")
            shutil.rmtree(settings.recovery_dir / f"v{vid}", ignore_errors=True)
            raise


def backfill_baseline(settings, conn: sqlite3.Connection, vid: int) -> str:
    """为"文件完好但缺可信依据"的版本补建依据（发布中断的补救）。

    只按磁盘现存字节建档，并要求两份文件都存在；不重算、不改版本行。
    """
    with exclusive_lock(settings):
        version = get_version(conn, vid)
        if version is None:
            raise IntegrityError("版本不存在")
        if get_baseline(conn, vid, "train") is not None:
            raise IntegrityError(f"版本 v{vid} 已有可信恢复依据")
        payload: dict[str, bytes] = {}
        missing = []
        for kind in INDEX_KINDS:
            rel = index_rel(version, kind)
            if not (settings.data_dir / rel).is_file():
                missing.append(INDEX_LABELS[kind])
                continue
            payload[kind] = (settings.data_dir / rel).read_bytes()
        if missing:
            raise IntegrityError("无法补建恢复依据：" + "、".join(missing) + "缺失")

        conn.execute("BEGIN IMMEDIATE")
        try:
            _snapshot_dir(settings, vid, payload)
            now = utcnow_str()
            for kind in INDEX_KINDS:
                data = payload[kind]
                rel = (settings.recovery_dir / f"v{vid}" / INDEX_FILENAMES[kind]).relative_to(
                    settings.data_dir
                )
                conn.execute(
                    "INSERT INTO index_baselines"
                    "(version_id, kind, sha256, size, rel_path, source, created_at)"
                    " VALUES (?,?,?,?,?,?,?)",
                    (vid, kind, sha256_bytes(data), len(data), str(rel), "backfill", now),
                )
            integrity = inspect_version(settings, conn, get_version(conn, vid))
            _set_verification(conn, vid, integrity)
            _event(conn, vid, "backfill", "ok", "disk", "按现存两份索引补建可信恢复依据")
            conn.execute("COMMIT")
            return f"版本 v{vid} 已按现存两份索引补建可信恢复依据"
        except BaseException:
            conn.execute("ROLLBACK")
            shutil.rmtree(settings.recovery_dir / f"v{vid}", ignore_errors=True)
            raise


# ---------------------------------------------------------------- 恢复

def _read_baseline_copy(settings, baseline) -> bytes:
    path = settings.data_dir / baseline["rel_path"]
    sha, size, err = file_digest(path)
    if sha is None or sha != baseline["sha256"] or size != baseline["size"]:
        raise IntegrityError("可信恢复副本缺失或损坏，无法从副本恢复；请上传原文件")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise IntegrityError(f"读取决信恢复副本失败：{exc}") from exc


def restore_index(
    settings,
    conn: sqlite3.Connection,
    vid: int,
    kind: str,
    *,
    data: bytes | None = None,
    source: str,
) -> str:
    """恢复一份索引。

    source:
      * upload   - data 为研究员上传的原文件，必须与可信摘要逐字节一致
      * baseline - 从可信恢复副本复制
      * repair   - 当前文件仍正确、可信副本损坏时，用当前文件重建副本
    """
    if kind not in INDEX_KINDS:
        raise IntegrityError("未知索引类型")

    with exclusive_lock(settings):
        version = get_version(conn, vid)
        if version is None:
            raise IntegrityError("版本不存在")
        baseline = get_baseline(conn, vid, kind)
        if baseline is None:
            raise IntegrityError(
                "该版本尚未建档，没有可信摘要，不能恢复。"
                "旧版请先按现存内容建档；已缺文件的旧版无法恢复。"
            )

        rel = index_rel(version, kind)
        target = settings.data_dir / rel

        # 先在锁外语义上准备好要写入的字节，再在事务内落盘。
        if source == "upload":
            if data is None:
                raise IntegrityError("请选择要恢复的索引文件")
            if sha256_bytes(data) != baseline["sha256"] or len(data) != baseline["size"]:
                msg = (
                    f"上传的{INDEX_LABELS[kind]}与可信恢复依据不一致"
                    f"（sha256 {sha256_bytes(data)[:12]}… ≠ {baseline['sha256'][:12]}…），"
                    "未写入任何文件"
                )
                conn.execute("BEGIN IMMEDIATE")
                _event(conn, vid, "recover", "rejected", source, f"{kind}: {msg}")
                conn.execute("COMMIT")
                raise IntegrityError(msg)
        elif source == "baseline":
            data = _read_baseline_copy(settings, baseline)
        elif source == "repair":
            actual_sha, _, _ = file_digest(target)
            if actual_sha != baseline["sha256"]:
                msg = "当前索引文件并非原字节，不能据此重建可信副本；请上传原文件恢复"
                conn.execute("BEGIN IMMEDIATE")
                _event(conn, vid, "recover", "rejected", source, f"{kind}: {msg}")
                conn.execute("COMMIT")
                raise IntegrityError(msg)
            try:
                data = target.read_bytes()
            except OSError as exc:
                raise IntegrityError(f"读取当前索引失败：{exc}") from exc
        else:
            raise IntegrityError("未知恢复方式")

        conn.execute("BEGIN IMMEDIATE")
        try:
            # 先把要写入的字节与"恢复后的预期状态"在事务内确定下来。
            # 写入本身对单个文件是原子的（两份中可能只有一份被替换），
            # 但下载放行条件是"两份都校验通过"，因此读者绝不会拿到半恢复的索引。
            if source == "repair":
                atomic_write(settings.data_dir / baseline["rel_path"], data)
            else:
                atomic_write(target, data)
                # 若可信副本本身残缺（上传恢复场景），顺带用原字节修复。
                copy_path = settings.data_dir / baseline["rel_path"]
                copy_sha, copy_size, _ = file_digest(copy_path)
                if copy_sha != baseline["sha256"] or copy_size != baseline["size"]:
                    atomic_write(copy_path, data)

            integrity = inspect_version(settings, conn, get_version(conn, vid))
            _set_verification(conn, vid, integrity)
            if integrity.ok:
                conn.execute(
                    "UPDATE versions SET recovered_at=? WHERE id=?", (utcnow_str(), vid)
                )
                status = "ok"
            else:
                status = "partial"
            _event(
                conn, vid, "recover", status, source,
                f"{kind}: {integrity.summary}",
            )
            conn.execute("COMMIT")
            if integrity.ok:
                return f"版本 v{vid} 的两份索引均已恢复为原字节，下载已恢复"
            return (
                f"{INDEX_LABELS[kind]}已恢复，但版本 v{vid} 另一份索引仍未通过校验："
                f"{integrity.summary}。在全部恢复前两份下载保持暂停。"
            )
        except BaseException:
            conn.execute("ROLLBACK")
            raise
