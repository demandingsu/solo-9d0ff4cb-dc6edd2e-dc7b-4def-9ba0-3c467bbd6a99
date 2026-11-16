"""野生动物红外相机照片索引服务（FastAPI + SQLite）。

核心保证
--------
* 每个含新照片的批次发布一个版本：train/val 两份索引 + 版本号。
* 版本不可变：索引文件写入 versions/v{N}/ 后永不修改，旧链接永远下载原内容。
* 发布是原子的：照片入库、索引文件落盘（tmp 目录 + os.rename）、
  versions 行与 active_version 指针在同一个 BEGIN IMMEDIATE 事务中提交；
  任一步失败整体回滚，现役版本不变。多进程/多请求交错上传由 SQLite
  写锁串行化，页面与两份下载始终对应同一完整版本。
* 幂等：同号同内容重投 -> 批次记为 noop，不产生新版本；
  同号异内容 -> 整批拒绝（rejected），库内不产生任何变化。
* 索引校验与恢复：每个新版本发布时，两份索引的 sha256 + 独立可信副本
  与版本行同事务留存（见 integrity.py）。任一索引缺失/被改动 -> 该版本
  两份下载一起暂停；恢复只接受与可信摘要逐字节一致的原文件，版本号、
  分组与训练/验证归属不变。旧版（上线前发布）两份都在时可按现存内容建档，
  缺文件的旧版提示无法恢复，绝不用当前照片表重算。
"""

from __future__ import annotations

import csv
import io
import os
import shutil
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from urllib.parse import urlencode

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response

from .grouping import Photo, assign_split, compute_groups
from .integrity import (
    INDEX_FILENAMES,
    INDEX_KINDS,
    INDEX_LABELS,
    STATUS_LABELS as INTEGRITY_LABELS,
    STATUS_OK,
    STATUS_UNBASELINED,
    IntegrityError,
    adopt_version,
    backfill_baseline,
    create_baseline,
    get_baseline,
    get_version,
    index_rel,
    inspect_version,
    record_verification,
    refresh_statuses,
    restore_index,
    sha256_bytes,
)

REQUIRED_COLUMNS = ("photo_no", "camera_no", "captured_at", "species", "rel_path")
INDEX_COLUMNS = ("photo_no", "camera_no", "captured_at", "species", "rel_path", "group_id")
MAX_UPLOAD_BYTES = 100 * 1024 * 1024

STATUS_LABELS = {
    "applied": "已发布",
    "noop": "无变化",
    "rejected": "已拒绝",
    "failed": "发布失败",
}


# ---------------------------------------------------------------- 配置

@dataclass(frozen=True)
class Settings:
    data_dir: Path
    val_percent: int = 20
    gap_seconds: int = 60
    split_salt: str = "wildlife-trap-v1"
    samples_dir: Path | None = None

    @property
    def db_path(self) -> Path:
        return self.data_dir / "app.db"

    @property
    def versions_dir(self) -> Path:
        return self.data_dir / "versions"

    @property
    def recovery_dir(self) -> Path:
        return self.data_dir / "recovery"

    @property
    def lock_path(self) -> Path:
        return self.data_dir / "integrity.lock"

    @classmethod
    def from_env(cls) -> "Settings":
        samples = os.environ.get("SAMPLES_DIR", "samples")
        samples_dir = Path(samples).resolve() if samples else None
        return cls(
            data_dir=Path(os.environ.get("DATA_DIR", "./data")).resolve(),
            val_percent=int(os.environ.get("VAL_PERCENT", "20")),
            gap_seconds=int(os.environ.get("GAP_SECONDS", "60")),
            split_salt=os.environ.get("SPLIT_SALT", "wildlife-trap-v1"),
            samples_dir=samples_dir if samples_dir and samples_dir.is_dir() else None,
        )


# ---------------------------------------------------------------- 时间

def utcnow_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ts_str(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_timestamp(raw: str) -> datetime:
    s = raw.strip()
    if not s:
        raise ValueError("空时间戳")
    dt = datetime.fromisoformat(s)  # 支持 "...Z" 与偏移；naive 视为 UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(microsecond=0)


def parse_stored_ts(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------- 数据库

SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uploaded_at TEXT NOT NULL,
    filename TEXT NOT NULL,
    rows_total INTEGER NOT NULL,
    rows_new INTEGER NOT NULL,
    rows_duplicate INTEGER NOT NULL,
    status TEXT NOT NULL,              -- applied | noop | rejected | failed
    message TEXT,
    version_id INTEGER
);
CREATE TABLE IF NOT EXISTS photos (
    photo_no TEXT PRIMARY KEY,
    camera_no TEXT NOT NULL,
    captured_at TEXT NOT NULL,         -- ISO8601 UTC, 秒精度
    species TEXT NOT NULL,
    rel_path TEXT NOT NULL,
    batch_id INTEGER NOT NULL REFERENCES batches(id)
);
CREATE TABLE IF NOT EXISTS versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    photo_count INTEGER NOT NULL,
    group_count INTEGER NOT NULL,
    train_count INTEGER NOT NULL,
    val_count INTEGER NOT NULL,
    train_path TEXT NOT NULL DEFAULT '',
    val_path TEXT NOT NULL DEFAULT '',
    integrity_status TEXT NOT NULL DEFAULT 'unbaselined',
    integrity_detail TEXT,
    recovered_at TEXT
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
-- 可信恢复依据：发布/建档时两份索引的 sha256、大小与独立副本。
CREATE TABLE IF NOT EXISTS index_baselines (
    version_id INTEGER NOT NULL REFERENCES versions(id),
    kind TEXT NOT NULL,                 -- train | val
    sha256 TEXT NOT NULL,
    size INTEGER NOT NULL,
    rel_path TEXT NOT NULL,             -- 独立副本（recovery/v{N}/...）
    source TEXT NOT NULL,               -- publish | adopt | backfill
    created_at TEXT NOT NULL,
    PRIMARY KEY (version_id, kind)
);
-- 最近一次显式校验结果（页面实时计算，不依赖此表）。
CREATE TABLE IF NOT EXISTS index_checks (
    version_id INTEGER NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,               -- ok | corrupt | missing
    sha256 TEXT,
    checked_at TEXT NOT NULL,
    detail TEXT,
    PRIMARY KEY (version_id, kind)
);
-- 校验/建档/恢复审计。
CREATE TABLE IF NOT EXISTS recovery_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version_id INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    action TEXT NOT NULL,               -- publish | verify | adopt | backfill | recover
    status TEXT NOT NULL,               -- ok | partial | rejected | failed
    source TEXT,                        -- publish | disk | upload | baseline | repair
    message TEXT
);
"""


def connect(settings: Settings) -> sqlite3.Connection:
    conn = sqlite3.connect(settings.db_path, timeout=15, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=15000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_storage(settings: Settings) -> None:
    settings.versions_dir.mkdir(parents=True, exist_ok=True)
    settings.recovery_dir.mkdir(parents=True, exist_ok=True)
    conn = connect(settings)
    try:
        conn.executescript(SCHEMA)
        _migrate_legacy_versions(conn)
        # 清理上次进程崩溃遗留的"目录已落盘、版本行未提交"的孤儿目录，
        # 否则重试发布会因目录已存在而失败。
        _cleanup_orphan_dirs(settings, conn)
        # 按磁盘现状刷新校验结果（只记录状态，不修改任何索引文件）。
        refresh_statuses(settings, conn)
    finally:
        conn.close()
    # 清理上次进程崩溃可能遗留的临时目录（完整版本目录永不动）
    for d in settings.versions_dir.iterdir():
        if d.is_dir() and d.name.startswith(".tmp-"):
            shutil.rmtree(d, ignore_errors=True)
    for d in settings.recovery_dir.iterdir():
        if d.is_dir() and d.name.startswith(".tmp-"):
            shutil.rmtree(d, ignore_errors=True)


def _migrate_legacy_versions(conn: sqlite3.Connection) -> None:
    """旧库升级：补 integrity_status 等列（默认 unbaselined，不自动信任）。"""
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(versions)")}
    if "integrity_status" not in cols:
        conn.execute("ALTER TABLE versions ADD COLUMN integrity_status TEXT NOT NULL DEFAULT 'unbaselined'")
    if "integrity_detail" not in cols:
        conn.execute("ALTER TABLE versions ADD COLUMN integrity_detail TEXT")
    if "recovered_at" not in cols:
        conn.execute("ALTER TABLE versions ADD COLUMN recovered_at TEXT")


def _cleanup_orphan_dirs(settings: Settings, conn: sqlite3.Connection) -> None:
    """删除没有对应 versions 行的 v{N} 目录（发布在 rename 后、提交前中断的遗留）。

    只在该版本既无 versions 行、也无 index_baselines 行时删除，绝不触碰有效版本。
    """
    known = {int(r["id"]) for r in conn.execute("SELECT id FROM versions")}
    known |= {
        int(r["version_id"])
        for r in conn.execute("SELECT DISTINCT version_id FROM index_baselines")
    }
    for root in (settings.versions_dir, settings.recovery_dir):
        try:
            entries = list(root.iterdir())
        except OSError:
            continue
        for d in entries:
            if not d.is_dir() or not d.name.startswith("v"):
                continue
            try:
                vid = int(d.name[1:])
            except ValueError:
                continue
            if vid not in known:
                shutil.rmtree(d, ignore_errors=True)


def get_active_version(conn: sqlite3.Connection) -> int | None:
    row = conn.execute("SELECT value FROM meta WHERE key='active_version'").fetchone()
    return int(row["value"]) if row else None


# ---------------------------------------------------------------- CSV 解析

@dataclass(frozen=True)
class CsvRow:
    photo_no: str
    camera_no: str
    captured_at: datetime
    species: str
    rel_path: str

    def content_tuple(self) -> tuple[str, str, str, str]:
        return (self.camera_no, ts_str(self.captured_at), self.species, self.rel_path)


def parse_csv(data: bytes) -> tuple[list[CsvRow], list[str]]:
    """解析并校验整份 CSV。任何行出错都返回错误列表（调用方整批拒绝）。"""
    errors: list[str] = []
    rows: list[CsvRow] = []
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return [], ["文件不是有效的 UTF-8 编码"]

    reader = csv.reader(io.StringIO(text))
    header = None
    for raw in reader:
        if raw and any(c.strip() for c in raw):
            header = raw
            break
    if header is None:
        return [], ["CSV 为空或缺少表头"]

    cols = {name.strip(): i for i, name in enumerate(header)}
    missing = [c for c in REQUIRED_COLUMNS if c not in cols]
    if missing:
        return [], [f"缺少必需列: {', '.join(missing)}（需要: {', '.join(REQUIRED_COLUMNS)}）"]

    seen: dict[str, CsvRow] = {}
    for lineno, raw in enumerate(reader, start=2):
        if not raw or all(not c.strip() for c in raw):
            continue

        def cell(name: str) -> str:
            i = cols[name]
            return raw[i].strip() if i < len(raw) else ""

        photo_no, camera_no = cell("photo_no"), cell("camera_no")
        ts_raw, species, rel_path = cell("captured_at"), cell("species"), cell("rel_path")

        row_errs: list[str] = []
        if not photo_no:
            row_errs.append("photo_no 为空")
        if not camera_no:
            row_errs.append("camera_no 为空")
        if not species:
            row_errs.append("species 为空")
        if not rel_path:
            row_errs.append("rel_path 为空")
        captured_at = None
        try:
            captured_at = parse_timestamp(ts_raw)
        except ValueError:
            row_errs.append(f"captured_at 无法解析: {ts_raw!r}（需 ISO8601，如 2026-03-01T08:00:00Z）")

        if row_errs:
            errors.append(f"第{lineno}行: " + "；".join(row_errs))
            continue

        row = CsvRow(photo_no, camera_no, captured_at, species, rel_path)
        if photo_no in seen:
            if seen[photo_no].content_tuple() != row.content_tuple():
                errors.append(f"第{lineno}行: photo_no {photo_no!r} 在文件内重复且内容不一致")
            continue  # 文件内完全重复的行视为同一照片
        seen[photo_no] = row
        rows.append(row)

    if not rows and not errors:
        errors.append("CSV 没有任何数据行")
    return rows, errors


# ---------------------------------------------------------------- 索引文件

def _write_index_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(INDEX_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def write_version_files(versions_dir: Path, vid: int, train_rows: list[dict], val_rows: list[dict]) -> Path:
    """把两份索引先写进临时目录，再原子 rename 成 versions/v{vid}/。

    rename 在同文件系统上是原子的：读者要么看不到该版本目录，
    要么看到包含两份完整索引的目录。
    """
    target = versions_dir / f"v{vid}"
    tmp = versions_dir / f".tmp-v{vid}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        _write_index_csv(tmp / "train_index.csv", train_rows)
        _write_index_csv(tmp / "val_index.csv", val_rows)
        if target.exists():
            raise FileExistsError(f"版本目录已存在: {target}")
        os.rename(tmp, target)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return target


# ---------------------------------------------------------------- 批次处理

def _record_batch(conn, filename, rows_total, rows_new, rows_dup, status, message, version_id=None) -> int:
    cur = conn.execute(
        "INSERT INTO batches(uploaded_at, filename, rows_total, rows_new, rows_duplicate, status, message, version_id)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (utcnow_str(), filename, rows_total, rows_new, rows_dup, status, message, version_id),
    )
    return int(cur.lastrowid)


def record_rejected_batch(settings: Settings, filename: str, rows_total: int, message: str) -> int:
    """CSV 解析/校验失败：不触碰照片数据，仅留审计记录。"""
    conn = connect(settings)
    try:
        conn.execute("BEGIN IMMEDIATE")
        bid = _record_batch(conn, filename, rows_total, 0, 0, "rejected", message)
        conn.execute("COMMIT")
        return bid
    finally:
        conn.close()


def process_batch(settings: Settings, filename: str, rows: list[CsvRow]) -> tuple[str, int, str]:
    """在单个写事务内完成：冲突检测 -> 入库 -> 归组划分 -> 索引落盘 -> 留存恢复依据 -> 切换现役版本。"""
    conn = connect(settings)
    published_dir: Path | None = None
    baseline_dir: Path | None = None
    try:
        conn.execute("BEGIN IMMEDIATE")

        existing = {
            r["photo_no"]: r
            for r in conn.execute("SELECT photo_no, camera_no, captured_at, species, rel_path FROM photos")
        }
        new_rows: list[CsvRow] = []
        conflicts: list[str] = []
        dup = 0
        for row in rows:
            ex = existing.get(row.photo_no)
            if ex is None:
                new_rows.append(row)
            elif (ex["camera_no"], ex["captured_at"], ex["species"], ex["rel_path"]) == row.content_tuple():
                dup += 1
            else:
                conflicts.append(row.photo_no)

        if conflicts:
            shown = ", ".join(sorted(conflicts)[:20])
            msg = f"整批拒绝：{len(conflicts)} 个照片号已存在但内容不一致（{shown}{' …' if len(conflicts) > 20 else ''}）"
            bid = _record_batch(conn, filename, len(rows), 0, dup, "rejected", msg)
            conn.execute("COMMIT")
            return "rejected", bid, msg

        if not new_rows:
            msg = f"批内 {dup} 张照片全部已存在且内容一致，未产生新版本"
            bid = _record_batch(conn, filename, len(rows), 0, dup, "noop", msg)
            conn.execute("COMMIT")
            return "noop", bid, msg

        batch_id = _record_batch(conn, filename, len(rows), len(new_rows), dup, "applied", None)
        conn.executemany(
            "INSERT INTO photos(photo_no, camera_no, captured_at, species, rel_path, batch_id) VALUES (?,?,?,?,?,?)",
            [(r.photo_no, r.camera_no, ts_str(r.captured_at), r.species, r.rel_path, batch_id) for r in new_rows],
        )

        # 基于全量照片重新归组（补传可能合并旧组）
        all_photos = [
            Photo(p["photo_no"], p["camera_no"], parse_stored_ts(p["captured_at"]), p["species"], p["rel_path"])
            for p in conn.execute("SELECT photo_no, camera_no, captured_at, species, rel_path FROM photos")
        ]
        groups = compute_groups(all_photos, settings.gap_seconds)

        train_rows: list[dict] = []
        val_rows: list[dict] = []
        for g in groups:
            target = val_rows if assign_split(g.group_id, settings.val_percent, settings.split_salt) == "val" else train_rows
            for p in g.photos:
                target.append({
                    "photo_no": p.photo_no,
                    "camera_no": p.camera_no,
                    "captured_at": ts_str(p.captured_at),
                    "species": p.species,
                    "rel_path": p.rel_path,
                    "group_id": g.group_id,
                })
        key = lambda r: (r["camera_no"], r["captured_at"], r["photo_no"])
        train_rows.sort(key=key)
        val_rows.sort(key=key)

        now = utcnow_str()
        cur = conn.execute(
            "INSERT INTO versions(created_at, photo_count, group_count, train_count, val_count) VALUES (?,?,?,?,?)",
            (now, len(all_photos), len(groups), len(train_rows), len(val_rows)),
        )
        vid = int(cur.lastrowid)

        # 索引文件落盘（tmp + rename）。失败 -> 异常 -> 整体回滚，现役版本不变。
        # 若上次发布在 rename 后、提交前中断，会遗留同号目录（该 vid 从未提交，
        # 因为本事务刚刚分配它）；先清掉，保证重试可以完成。
        orphan = settings.versions_dir / f"v{vid}"
        if orphan.exists():
            shutil.rmtree(orphan, ignore_errors=True)
        published_dir = write_version_files(settings.versions_dir, vid, train_rows, val_rows)
        rel = Path("versions") / f"v{vid}"
        train_rel = str(rel / "train_index.csv")
        val_rel = str(rel / "val_index.csv")
        conn.execute(
            "UPDATE versions SET train_path=?, val_path=? WHERE id=?",
            (train_rel, val_rel, vid),
        )
        # 可信恢复依据：两份索引的独立副本 + sha256，与版本在同一事务内成立。
        baseline_dir = settings.recovery_dir / f"v{vid}"
        if baseline_dir.exists():
            shutil.rmtree(baseline_dir, ignore_errors=True)
        create_baseline(settings, conn, vid, train_rel, val_rel, source="publish")
        conn.execute(
            "INSERT INTO meta(key, value) VALUES ('active_version', ?)"
            " ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(vid),),
        )
        conn.execute("UPDATE batches SET version_id=? WHERE id=?", (vid, batch_id))
        msg = f"已发布版本 v{vid}：新增 {len(new_rows)} 张（重复 {dup} 张），共 {len(all_photos)} 张 / {len(groups)} 组，训练 {len(train_rows)} / 验证 {len(val_rows)}；两份索引已留存可信恢复依据"
        conn.execute("UPDATE batches SET message=? WHERE id=?", (msg, batch_id))
        conn.execute("COMMIT")
        return "applied", batch_id, msg

    except BaseException as exc:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        if published_dir is not None:  # 回滚后版本目录成为孤儿，安全清理（写锁保证只有本事务在写）
            shutil.rmtree(published_dir, ignore_errors=True)
        if baseline_dir is not None:  # 回滚后可信副本目录同样成为孤儿
            shutil.rmtree(baseline_dir, ignore_errors=True)
        try:  # 失败审计（独立小事务，失败不影响主流程）
            conn.execute("BEGIN IMMEDIATE")
            _record_batch(conn, filename, len(rows), 0, 0, "failed", f"发布失败，现役版本未变：{exc}"[:500])
            conn.execute("COMMIT")
        except sqlite3.Error:
            pass
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------- 页面渲染

PAGE_CSS = """
body{font-family:-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,"Noto Sans SC",sans-serif;margin:0;background:#f5f6f8;color:#222}
header{background:#2f5233;color:#fff;padding:14px 24px}
header h1{margin:0;font-size:20px}
header a{color:#fff;text-decoration:none}
main{max-width:1080px;margin:24px auto;padding:0 16px}
.card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:16px 20px;margin-bottom:20px}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{border:1px solid #ddd;padding:6px 10px;text-align:left;vertical-align:top}
th{background:#eef3ee}
.badge{display:inline-block;background:#2f5233;color:#fff;border-radius:4px;padding:1px 8px;font-size:12px}
.flash-ok{background:#e6f4e6;border:1px solid #7ab87a;color:#1d5c1d}
.flash-err{background:#fde8e8;border:1px solid #e08a8a;color:#8a1f1f}
.flash-info{background:#e8f0fe;border:1px solid #8aa8e0;color:#1f3d8a}
.flash{border-radius:6px;padding:10px 14px;margin-bottom:16px;font-size:14px}
code,pre{background:#f0f0f0;border-radius:4px}
code{padding:1px 5px}
pre{padding:10px;overflow:auto;font-size:13px}
.btn{display:inline-block;background:#2f5233;color:#fff;border:none;border-radius:5px;padding:7px 16px;font-size:14px;cursor:pointer;text-decoration:none}
.btn.secondary{background:#5a6b5d}
.muted{color:#666;font-size:13px}
ol.steps li{margin-bottom:6px}
.tag{display:inline-block;border-radius:4px;padding:1px 7px;font-size:12px;white-space:nowrap}
.tag-ok{background:#e6f4e6;border:1px solid #7ab87a;color:#1d5c1d}
.tag-bad{background:#fde8e8;border:1px solid #e08a8a;color:#8a1f1f}
.paused{color:#8a1f1f;font-weight:600}
form.inline{display:inline-block;margin:2px 6px 2px 0}
form.inline input[type=file]{max-width:180px;font-size:12px;display:block;margin-bottom:4px}
.callout{border:1px solid #8aa8e0;background:#eef4ff;border-radius:6px;padding:10px 16px;margin-bottom:20px;font-size:14px}
.callout-err{border-color:#e08a8a;background:#fdeeee}
"""


def page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)} · 红外相机照片索引</title>
<style>{PAGE_CSS}</style></head>
<body><header><h1><a href="/">野生动物红外相机照片索引</a></h1></header>
<main>{body}</main></body></html>"""


def flash_html(msg: str, kind: str) -> str:
    if not msg:
        return ""
    kind = kind if kind in ("ok", "err", "info") else "info"
    return f'<div class="flash flash-{kind}">{escape(msg)}</div>'


def integrity_badge(status: str) -> str:
    cls = "ok" if status in (STATUS_OK, STATUS_UNBASELINED) else "bad"
    lab = INTEGRITY_LABELS.get(status, "下载已暂停")
    return f'<span class="tag tag-{cls}">{escape(lab)}</span>'


def home_page(settings: Settings, msg: str, kind: str) -> str:
    conn = connect(settings)
    try:
        active_id = get_active_version(conn)
        versions = conn.execute("SELECT * FROM versions ORDER BY id DESC").fetchall()
        batches = conn.execute("SELECT * FROM batches ORDER BY id DESC LIMIT 50").fetchall()
        states = {int(v["id"]): inspect_version(settings, conn, v) for v in versions}
    finally:
        conn.close()

    # 现役版本横幅：校验未通过的版本不发下载链接（两份一起暂停）。
    if active_id is not None:
        active = next((v for v in versions if v["id"] == active_id), None)
        if active is None:
            banner = '<div class="card"><p>现役版本记录缺失。</p></div>'
        else:
            st = states[active_id]
            if st.downloads_allowed:
                links = (
                    f'<p><a class="btn" href="/versions/{active_id}/files/train_index.csv">下载训练索引</a> '
                    f'<a class="btn" href="/versions/{active_id}/files/val_index.csv">下载验证索引</a> '
                    f'<a class="btn secondary" href="/versions/{active_id}">查看分组</a></p>'
                )
            else:
                links = (
                    f'<p><a class="btn secondary" href="/versions/{active_id}">查看并处理索引异常</a></p>'
                )
            banner = (
                f'<div class="card"><h2 style="margin-top:0">现役版本 <span class="badge">v{active_id}</span> '
                f"{integrity_badge(st.status)}</h2>"
                f"<p>照片 {active['photo_count']} 张 · 出没事件组 {active['group_count']} 个 · "
                f"训练 {active['train_count']} / 验证 {active['val_count']} · 发布于 {active['created_at']} UTC</p>"
                f'<p class="muted">{escape(st.summary)}</p>{links}</div>'
            )
    else:
        banner = '<div class="card"><p>尚未发布任何版本，请上传第一批 CSV。</p></div>'

    # 上传表单
    upload = f"""
<div class="card"><h2 style="margin-top:0">上传照片 CSV 并发布新版本</h2>
<form action="/upload" method="post" enctype="multipart/form-data">
<p><input type="file" name="file" accept=".csv,text/csv" required>
<button class="btn" type="submit">上传并发布</button></p>
</form>
<p class="muted">必需列：<code>{'</code> <code>'.join(REQUIRED_COLUMNS)}</code>；
时间为 UTC（ISO8601，如 <code>2026-03-01T08:00:00Z</code>）。
同一相机相邻 ≤{settings.gap_seconds} 秒的照片传递归组；按组稳定划分（验证集比例 {settings.val_percent}%）。
同号同内容重投不产生新版本；同号异内容整批拒绝。</p></div>"""

    # 样例
    samples_html = ""
    if settings.samples_dir:
        files = sorted(p.name for p in settings.samples_dir.glob("*.csv"))
        if files:
            links = " ".join(f'<li><a href="/samples/{escape(n)}">{escape(n)}</a></li>' for n in files)
            samples_html = f"""
<div class="card"><h2 style="margin-top:0">跨批补传演示（浏览器可操作）</h2>
<ol class="steps">
<li>下载样例 CSV：<ul>{links}</ul></li>
<li>上传 <code>batch1_base.csv</code> → 发布 v1（CAM01 的两段出没因间隔 70 秒分为两组）。</li>
<li>上传 <code>batch2_backfill.csv</code> → 发布 v2：补传的 P0007（08:01:05）与前后两组各相隔 35 秒，
把旧两组合并成一组，全库重新归组；P0008 加入 CAM02 旧组。</li>
<li>上传 <code>batch3_duplicate.csv</code>（与 batch2 完全相同）→ 不产生新版本。</li>
<li>上传 <code>batch4_conflict.csv</code>（P0001 内容被改动）→ 整批拒绝，现役版本保持 v2。</li>
<li>随时打开 v1 的下载链接，内容保持最初发布时的原样。</li>
</ol></div>"""

    # 版本表
    if versions:
        vrows = []
        for v in versions:
            vid = int(v["id"])
            st = states[vid]
            active_badge = ' <span class="badge">现役</span>' if vid == active_id else ""
            if st.downloads_allowed:
                downloads = (
                    f'<a href="/versions/{vid}/files/train_index.csv">train</a> · '
                    f'<a href="/versions/{vid}/files/val_index.csv">val</a>'
                )
            else:
                downloads = '<span class="paused">下载已暂停</span>'
            vrows.append(
                f"<tr><td><a href=\"/versions/{vid}\">v{vid}</a>{active_badge}</td>"
                f"<td>{v['created_at']}</td><td>{v['photo_count']}</td><td>{v['group_count']}</td>"
                f"<td>{v['train_count']}</td><td>{v['val_count']}</td>"
                f"<td>{integrity_badge(st.status)}</td><td>{downloads}</td></tr>"
            )
        versions_html = (
            '<div class="card"><h2 style="margin-top:0">版本历史（旧版本链接内容不变）</h2>'
            "<table><tr><th>版本</th><th>发布时间(UTC)</th><th>照片数</th><th>组数</th>"
            "<th>训练</th><th>验证</th><th>索引校验</th><th>索引下载</th></tr>"
            + "".join(vrows) + "</table></div>"
        )
    else:
        versions_html = ""

    # 批次表
    if batches:
        brows = []
        for b in batches:
            label = STATUS_LABELS.get(b["status"], b["status"])
            ver = f'<a href="/versions/{b["version_id"]}">v{b["version_id"]}</a>' if b["version_id"] else "—"
            brows.append(
                f'<tr><td><a href="/batches/{b["id"]}">#{b["id"]}</a></td><td>{b["uploaded_at"]}</td>'
                f"<td>{escape(b['filename'])}</td><td>{label}</td>"
                f"<td>{b['rows_total']}</td><td>{b['rows_new']}</td><td>{b['rows_duplicate']}</td><td>{ver}</td></tr>"
            )
        batches_html = (
            '<div class="card"><h2 style="margin-top:0">批次记录</h2>'
            "<table><tr><th>批次</th><th>时间(UTC)</th><th>文件名</th><th>状态</th>"
            "<th>总行数</th><th>新增</th><th>重复</th><th>产生版本</th></tr>" + "".join(brows) + "</table></div>"
        )
    else:
        batches_html = ""

    return page("首页", flash_html(msg, kind) + banner + upload + samples_html + versions_html + batches_html)


def version_detail_page(settings: Settings, vid: int, msg: str = "", kind: str = "info") -> str:
    conn = connect(settings)
    try:
        v = get_version(conn, vid)
        active_id = get_active_version(conn)
        if v is None:
            raise HTTPException(404, "版本不存在")
        integrity = inspect_version(settings, conn, v)
        events = conn.execute(
            "SELECT * FROM recovery_events WHERE version_id=? ORDER BY id DESC LIMIT 20",
            (vid,),
        ).fetchall()
        photo_count = conn.execute("SELECT COUNT(*) AS n FROM photos").fetchone()["n"]
        # 发布时若恢复依据没写成，提示可补建（只按现存字节，不重算）。
        needs_retry = (
            active_id is not None and get_baseline(conn, active_id, "train") is None
        )
    finally:
        conn.close()

    groups: dict[str, dict] = {}
    read_errors: list[str] = []
    for split, kind_name in (("train", "train"), ("val", "val")):
        path = settings.data_dir / index_rel(v, kind_name)
        try:
            with open(path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    # 异常索引可能含残缺行；展示层跳过，不影响校验/暂停判定。
                    if not row.get("group_id") or not row.get("captured_at") or not row.get("photo_no"):
                        continue
                    g = groups.setdefault(row["group_id"], {"split": split, "rows": []})
                    g["rows"].append(row)
        except (OSError, KeyError, csv.Error) as exc:
            read_errors.append(f"{INDEX_LABELS[kind_name]}无法用于展示分组：{exc}")

    grows = []
    sortable = [gid for gid in groups if groups[gid]["rows"]]
    for gid in sorted(sortable, key=lambda g: (min(r["captured_at"] for r in groups[g]["rows"]), g)):
        g = groups[gid]
        times = sorted(r["captured_at"] for r in g["rows"])
        members = ", ".join(r["photo_no"] for r in sorted(g["rows"], key=lambda r: (r["captured_at"], r["photo_no"])))
        split_label = "验证" if g["split"] == "val" else "训练"
        grows.append(
            f"<tr><td>{escape(gid)}</td><td>{escape(g['rows'][0]['camera_no'])}</td>"
            f"<td>{times[0]}</td><td>{times[-1]}</td><td>{len(g['rows'])}</td>"
            f"<td>{split_label}</td><td>{escape(members)}</td></tr>"
        )

    # ---------------------------------------------------- 校验与恢复面板
    rows_html = []
    for kind in INDEX_KINDS:
        st = integrity.states[kind]
        recorded = st.recorded_sha256 or "—"
        actual = st.actual_sha256 or "—"
        actions = []
        if st.can_upload and not st.ok:
            actions.append(
                f'<form class="inline" action="/versions/{vid}/recover" method="post" enctype="multipart/form-data">'
                f'<input type="hidden" name="kind" value="{kind}">'
                f'<input type="file" name="file" accept=".csv,text/csv" required>'
                f'<button class="btn" type="submit">上传原文件恢复</button></form>'
            )
        if st.can_restore_copy and not st.ok:
            actions.append(
                f'<form class="inline" action="/versions/{vid}/recover" method="post">'
                f'<input type="hidden" name="kind" value="{kind}">'
                f'<input type="hidden" name="source" value="baseline">'
                f'<button class="btn secondary" type="submit">从可信副本恢复</button></form>'
            )
        if st.can_repair_copy and not st.ok:
            actions.append(
                f'<form class="inline" action="/versions/{vid}/recover" method="post">'
                f'<input type="hidden" name="kind" value="{kind}">'
                f'<input type="hidden" name="source" value="repair">'
                f'<button class="btn secondary" type="submit">重建可信副本</button></form>'
            )
        if not actions:
            actions.append(
                '<span class="muted">无需恢复</span>' if st.ok else '<span class="muted">无可用操作</span>'
            )
        rows_html.append(
            f"<tr><td>{escape(st.label)}<br><code>{escape(st.filename)}</code></td>"
            f"<td>{integrity_badge(st.status)}<br><span class=\"muted\">{escape(st.detail)}</span></td>"
            f"<td><code title=\"可信摘要\">{recorded[:16]}{'…' if recorded != '—' else ''}</code></td>"
            f"<td><code title=\"当前磁盘内容摘要\">{actual[:16]}{'…' if actual != '—' else ''}</code></td>"
            f"<td>{''.join(actions)}</td></tr>"
        )

    if integrity.ok:
        dl = (
            f'<p><a class="btn" href="/versions/{vid}/files/train_index.csv">下载训练索引</a> '
            f'<a class="btn" href="/versions/{vid}/files/val_index.csv">下载验证索引</a></p>'
        )
        paused_note = ""
    elif integrity.status == STATUS_UNBASELINED:
        dl = (
            f'<p><a class="btn" href="/versions/{vid}/files/train_index.csv">下载训练索引</a> '
            f'<a class="btn" href="/versions/{vid}/files/val_index.csv">下载验证索引</a></p>'
        )
        paused_note = (
            '<p class="muted">旧版尚未建档：当前可继续下载，但在此之前无法发现文件改动。'
            "建议按现存内容建档。</p>"
        )
    else:
        dl = '<p><span class="paused">该版本两份下载均已暂停（任一份异常即同时停用）</span></p>'
        paused_note = (
            '<p class="muted">说明：恢复只写回与可信摘要逐字节一致的原文件；版本号、分组与训练/验证归属'
            "均来自原索引，不会用当前照片表重算。</p>"
        )

    if integrity.can_adopt:
        adopt_form = f"""
<div class="callout">
<p><strong>这是一个旧版本</strong>：发布时尚未留存可信恢复依据。两份索引目前都可读取，
可按现存内容建档；建档后任何改动都会被校验发现。</p>
<form action="/versions/{vid}/adopt" method="post">
<button class="btn" type="submit">按现存内容建档</button></form>
</div>"""
    elif all(st.recorded_sha256 is None for st in integrity.states.values()):
        # 无任何可信摘要：只要有一份缺失，就不能建档（也不能重算）。
        adopt_form = """
<div class="callout callout-err">
<p><strong>该旧版本无法建档恢复</strong>：至少一份索引文件缺失，且发布时未留存可信恢复依据。
不会根据当前照片表重算（那样无法保证原版本号、分组与训练/验证归属）。</p>
</div>"""
    else:
        adopt_form = ""

    evrows = "".join(
        f"<tr><td>{e['created_at']}</td><td>{escape(e['action'])}</td>"
        f"<td>{escape(e['status'])}</td><td>{escape(e['source'] or '—')}</td>"
        f"<td>{escape(e['message'] or '')}</td></tr>"
        for e in events
    )
    events_html = (
        '<div class="card"><h3 style="margin-top:0">校验与恢复记录</h3>'
        "<table><tr><th>时间(UTC)</th><th>操作</th><th>结果</th><th>来源</th><th>说明</th></tr>"
        f"{evrows}</table></div>"
        if events else ""
    )

    group_hint = ""
    if read_errors:
        group_hint = "".join(f'<p class="muted">{escape(e)}</p>' for e in read_errors)
    else:
        group_hint = (
            f'<p class="muted">当前照片库共 {photo_count} 张；本页分组来自该版本索引原文件。</p>'
        )

    retry_html = ""
    if needs_retry:
        retry_html = f"""
<div class="callout">
<p><strong>现役版本 v{active_id} 尚无可信恢复依据</strong>（上次发布可能在中途中断）。
若两份索引文件仍在，可按现存字节补建恢复依据；不会新增照片，也不会改变版本号、分组与训练/验证归属。</p>
<form action="/versions/{active_id}/backfill" method="post">
<button class="btn" type="submit">按现存文件补建可信恢复依据</button></form>
</div>"""

    active_badge = ' <span class="badge">现役</span>' if vid == active_id else ""
    body = f"""
<div class="card"><h2 style="margin-top:0">版本 v{vid}{active_badge} {integrity_badge(integrity.status)}</h2>
<p>发布于 {v['created_at']} UTC · 照片 {v['photo_count']} 张 · 组 {v['group_count']} 个 ·
训练 {v['train_count']} / 验证 {v['val_count']}</p>
<p class="muted">{escape(integrity.summary)}{(' · 最近恢复于 ' + escape(v['recovered_at'] or '')) if v['recovered_at'] else ''}</p>
{dl}
<p><a class="btn secondary" href="/">返回首页</a></p></div>
{retry_html}
<div class="card"><h3 style="margin-top:0">索引校验与恢复</h3>
<table><tr><th>索引</th><th>状态</th><th>可信摘要</th><th>当前文件摘要</th><th>恢复操作</th></tr>
{''.join(rows_html)}</table>
{paused_note}
<form class="inline" action="/versions/{vid}/verify" method="post">
<button class="btn secondary" type="submit">重新校验</button></form>
</div>
{adopt_form}
<div class="card"><h3 style="margin-top:0">出没事件分组（group_id = 组内最早照片号）</h3>
{group_hint}
<table><tr><th>group_id</th><th>相机</th><th>开始(UTC)</th><th>结束(UTC)</th><th>张数</th><th>划分</th><th>成员照片</th></tr>
{''.join(grows)}</table></div>
{events_html}"""
    return page(f"版本 v{vid}", flash_html(msg, kind) + body)


def batch_detail_page(settings: Settings, bid: int) -> str:
    conn = connect(settings)
    try:
        b = conn.execute("SELECT * FROM batches WHERE id=?", (bid,)).fetchone()
    finally:
        conn.close()
    if b is None:
        raise HTTPException(404, "批次不存在")
    label = STATUS_LABELS.get(b["status"], b["status"])
    ver = f'<a href="/versions/{b["version_id"]}">v{b["version_id"]}</a>' if b["version_id"] else "—"
    photos = []
    if b["status"] == "applied":
        conn = connect(settings)
        try:
            photos = conn.execute(
                "SELECT photo_no, camera_no, captured_at, species, rel_path FROM photos WHERE batch_id=? ORDER BY camera_no, captured_at, photo_no",
                (bid,),
            ).fetchall()
        finally:
            conn.close()
    prows = "".join(
        f"<tr><td>{escape(p['photo_no'])}</td><td>{escape(p['camera_no'])}</td><td>{p['captured_at']}</td>"
        f"<td>{escape(p['species'])}</td><td>{escape(p['rel_path'])}</td></tr>"
        for p in photos
    )
    photos_html = (
        f'<div class="card"><h2 style="margin-top:0">本批入库照片（{len(photos)}）</h2>'
        "<table><tr><th>照片号</th><th>相机</th><th>拍摄时间(UTC)</th><th>物种</th><th>原图相对路径</th></tr>"
        f"{prows}</table></div>"
        if photos else ""
    )
    body = f"""
<div class="card"><h2 style="margin-top:0">批次 #{b['id']}</h2>
<table>
<tr><th>时间(UTC)</th><td>{b['uploaded_at']}</td></tr>
<tr><th>文件名</th><td>{escape(b['filename'])}</td></tr>
<tr><th>状态</th><td>{label}</td></tr>
<tr><th>总行数 / 新增 / 重复</th><td>{b['rows_total']} / {b['rows_new']} / {b['rows_duplicate']}</td></tr>
<tr><th>产生版本</th><td>{ver}</td></tr>
<tr><th>详情</th><td><pre>{escape(b['message'] or '—')}</pre></td></tr>
</table>
<p><a class="btn secondary" href="/">返回首页</a></p></div>
{photos_html}"""
    return page(f"批次 #{bid}", body)


# ---------------------------------------------------------------- 应用

def create_app(settings: Settings) -> FastAPI:
    init_storage(settings)
    app = FastAPI(title="野生动物红外相机照片索引", docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/", response_class=HTMLResponse)
    def home(msg: str = "", kind: str = "info") -> str:
        return home_page(settings, msg, kind)

    @app.post("/upload")
    def upload(file: UploadFile = File(...)):
        filename = file.filename or "upload.csv"
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            return RedirectResponse("/?" + urlencode({"kind": "err", "msg": "文件超过 100MB 限制"}), 303)

        rows, errors = parse_csv(data)
        if errors:
            bid = record_rejected_batch(settings, filename, len(rows), "整批拒绝：\n" + "\n".join(errors[:50]))
            return RedirectResponse(
                "/?" + urlencode({"kind": "err", "msg": f"批次 #{bid} 被拒绝：{errors[0]}" + (" …" if len(errors) > 1 else "")}),
                303,
            )

        try:
            status, batch_id, message = process_batch(settings, filename, rows)
        except Exception as exc:  # 发布失败：事务已回滚，现役版本不变
            detail = f"：{exc}" if str(exc) else ""
            return RedirectResponse(
                "/?" + urlencode({"kind": "err", "msg": f"发布失败，现役版本未变{detail}"}), 303
            )
        kind = {"applied": "ok", "noop": "info", "rejected": "err"}[status]
        return RedirectResponse("/?" + urlencode({"kind": kind, "msg": message}), 303)

    @app.get("/versions/{vid}", response_class=HTMLResponse)
    def version_detail(vid: int, msg: str = "", kind: str = "info") -> str:
        return version_detail_page(settings, vid, msg, kind)

    def _version_redirect(vid: int, msg: str, kind: str) -> RedirectResponse:
        return RedirectResponse(
            f"/versions/{vid}?" + urlencode({"kind": kind, "msg": msg}), 303
        )

    @app.post("/versions/{vid}/verify")
    def verify_version(vid: int):
        conn = connect(settings)
        try:
            try:
                integrity = record_verification(settings, conn, vid)
            except IntegrityError as exc:
                raise HTTPException(404, str(exc)) from exc
        finally:
            conn.close()
        if integrity.status == STATUS_UNBASELINED:
            return _version_redirect(
                vid, "该旧版本尚未建档：两份索引可读，请按现存内容建档后再校验", "info"
            )
        if not integrity.ok:
            return _version_redirect(vid, "校验未通过：" + integrity.summary, "err")
        return _version_redirect(vid, "校验通过：两份索引均与原发布字节一致", "ok")

    @app.post("/versions/{vid}/adopt")
    def adopt(vid: int):
        conn = connect(settings)
        try:
            try:
                msg = adopt_version(settings, conn, vid)
            except IntegrityError as exc:
                return _version_redirect(vid, str(exc), "err")
        finally:
            conn.close()
        return _version_redirect(vid, msg, "ok")

    @app.post("/versions/{vid}/backfill")
    def backfill(vid: int):
        conn = connect(settings)
        try:
            try:
                msg = backfill_baseline(settings, conn, vid)
            except (IntegrityError, OSError) as exc:
                return _version_redirect(vid, str(exc), "err")
        finally:
            conn.close()
        return _version_redirect(vid, msg, "ok")

    @app.post("/versions/{vid}/recover")
    def recover(
        vid: int,
        kind: str = Form(...),
        source: str = Form("upload"),
        file: UploadFile | None = File(None),
    ):
        if kind not in INDEX_KINDS:
            return _version_redirect(vid, "未知索引类型", "err")
        data = None
        src = source
        if source == "upload":
            if file is None:
                return _version_redirect(vid, "请选择要上传恢复的索引文件", "err")
            data = file.file.read(MAX_UPLOAD_BYTES + 1)
            if len(data) > MAX_UPLOAD_BYTES:
                return _version_redirect(vid, "文件超过 100MB 限制", "err")
        elif source not in ("baseline", "repair"):
            return _version_redirect(vid, "未知恢复方式", "err")

        conn = connect(settings)
        try:
            try:
                msg = restore_index(settings, conn, vid, kind, data=data, source=src)
            except IntegrityError as exc:
                return _version_redirect(vid, str(exc), "err")
            except OSError as exc:
                return _version_redirect(
                    vid, f"恢复写入失败，未提供半恢复的索引；可重试：{exc}", "err"
                )
        finally:
            conn.close()
        kindcls = "ok"
        conn = connect(settings)
        try:
            v = get_version(conn, vid)
            if v is not None and not inspect_version(settings, conn, v).ok:
                kindcls = "info"
        finally:
            conn.close()
        return _version_redirect(vid, msg, kindcls)

    @app.get("/versions/{vid}/files/{name}")
    def version_file(vid: int, name: str):
        if name not in INDEX_FILENAMES.values():
            raise HTTPException(404, "未知文件")
        conn = connect(settings)
        try:
            v = get_version(conn, vid)
            if v is None:
                raise HTTPException(404, "版本不存在")
            integrity = inspect_version(settings, conn, v)
        finally:
            conn.close()
        which = "train" if name == "train_index.csv" else "val"
        if not integrity.downloads_allowed:
            # 任一份异常 -> 两份一起暂停，绝不提供可能已被改动的索引。
            raise HTTPException(409, "该版本索引校验未通过，两份下载均已暂停：" + integrity.summary)
        path = settings.data_dir / index_rel(v, which)
        # 读入内存并按刚校验过的摘要再确认一次，避免校验后文件被替换的竞态。
        try:
            data = path.read_bytes()
        except OSError:
            raise HTTPException(409, "索引文件缺失，下载已暂停")
        expected = integrity.states[which].recorded_sha256
        if expected is not None and sha256_bytes(data) != expected:
            raise HTTPException(409, "索引在读取过程中被改动，下载已暂停；请重新校验并恢复")
        return Response(
            content=data,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="v{vid}_{name}"'},
        )

    @app.get("/batches/{bid}", response_class=HTMLResponse)
    def batch_detail(bid: int) -> str:
        return batch_detail_page(settings, bid)

    if settings.samples_dir:
        samples_root = settings.samples_dir

        @app.get("/samples/{name}")
        def sample_file(name: str):
            path = (samples_root / name).resolve()
            if not path.is_file() or samples_root not in path.parents:
                raise HTTPException(404, "样例不存在")
            return FileResponse(path, media_type="text/csv", filename=name)

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}

    return app


app = create_app(Settings.from_env())
