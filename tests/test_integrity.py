"""索引校验与恢复的端到端测试：损坏检测、双份下载暂停、原字节恢复、
旧版建档、缺失文件不可恢复、恢复中断/重试、与补传发布交错。"""
import csv
import io
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.integrity as integrity_mod
import app.main as main
from app.main import Settings, create_app


def make_csv(rows):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["photo_no", "camera_no", "captured_at", "species", "rel_path"])
    w.writerows(rows)
    return buf.getvalue().encode()


BATCH1 = make_csv([
    ["P0002", "CAM01", "2026-03-01T08:00:00Z", "wild_boar", "cam01/P0002.jpg"],
    ["P0003", "CAM01", "2026-03-01T08:00:30Z", "wild_boar", "cam01/P0003.jpg"],
    ["P0005", "CAM01", "2026-03-01T08:01:40Z", "roe_deer", "cam01/P0005.jpg"],
    ["P0006", "CAM01", "2026-03-01T08:02:10Z", "roe_deer", "cam01/P0006.jpg"],
    ["P0001", "CAM02", "2026-03-01T08:00:10Z", "red_fox", "cam02/P0001.jpg"],
    ["P0004", "CAM02", "2026-03-01T09:30:00Z", "red_fox", "cam02/P0004.jpg"],
])

BACKFILL = make_csv([
    ["P0007", "CAM01", "2026-03-01T08:01:05Z", "wild_boar", "cam01/P0007.jpg"],
    ["P0008", "CAM02", "2026-03-01T09:30:40Z", "red_fox", "cam02/P0008.jpg"],
    ["P0009", "CAM03", "2026-03-01T12:00:00Z", "sika_deer", "cam03/P0009.jpg"],
])


@pytest.fixture()
def env(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", val_percent=20, gap_seconds=60)
    app = create_app(settings)
    client = TestClient(app, follow_redirects=True)
    return client, settings


def upload(client, data, name="batch.csv"):
    return client.post("/upload", files={"file": (name, data, "text/csv")})


def idx_path(settings, vid, kind):
    name = "train_index.csv" if kind == "train" else "val_index.csv"
    return settings.data_dir / "versions" / f"v{vid}" / name


def recovery_path(settings, vid, kind):
    name = "train_index.csv" if kind == "train" else "val_index.csv"
    return settings.data_dir / "recovery" / f"v{vid}" / name


def download(client, vid, kind):
    name = "train_index.csv" if kind == "train" else "val_index.csv"
    return client.get(f"/versions/{vid}/files/{name}")


def recover(client, vid, kind, source, data=None):
    files = None
    form = {"kind": kind, "source": source}
    if source == "upload":
        files = {"file": (f"{kind}_index.csv", data if data is not None else b"", "text/csv")}
    return client.post(f"/versions/{vid}/recover", data=form, files=files)


def plain_client(settings):
    """重新创建应用（模拟进程重启），用于验证状态不依赖进程内缓存。"""
    return TestClient(create_app(settings), follow_redirects=True)


# ---------------------------------------------------------------- 发布留存依据

def test_publish_keeps_two_trusted_copies_and_hashes(env):
    client, settings = env
    upload(client, BATCH1, "b1.csv")

    for kind in ("train", "val"):
        assert idx_path(settings, 1, kind).is_file()
        assert recovery_path(settings, 1, kind).is_file()
        assert recovery_path(settings, 1, kind).read_bytes() == idx_path(settings, 1, kind).read_bytes()

    conn = sqlite3.connect(settings.db_path)
    rows = conn.execute(
        "SELECT kind, sha256, size, source FROM index_baselines WHERE version_id=1 ORDER BY kind"
    ).fetchall()
    conn.close()
    assert [r[0] for r in rows] == ["train", "val"]
    assert all(r[2] > 0 and len(r[1]) == 64 and r[3] == "publish" for r in rows)

    body = client.get("/versions/1").text
    assert "校验通过" in body
    assert "上传原文件恢复" not in body  # 完好版本不提供恢复入口
    assert download(client, 1, "train").status_code == 200
    assert download(client, 1, "val").status_code == 200


# ---------------------------------------------------------------- 损坏 -> 双暂停

@pytest.mark.parametrize("kind", ["train", "val"])
def test_corruption_pauses_both_downloads_and_restores_exact_bytes(env, kind):
    client, settings = env
    upload(client, BATCH1, "b1.csv")
    upload(client, BACKFILL, "b2.csv")

    train_before = download(client, 2, "train").content
    val_before = download(client, 2, "val").content
    detail_before = client.get("/versions/2").text  # 含分组明细

    target = idx_path(settings, 2, kind)
    target.write_bytes(target.read_bytes() + b"tampered-row\n")

    # 任一份异常 -> 两份都暂停，不论损坏的是哪一份
    assert download(client, 2, "train").status_code == 409
    assert download(client, 2, "val").status_code == 409
    body = client.get("/versions/2").text
    assert "内容被改动" in body and "两份下载均已暂停" in body
    # 其他完好版本照常下载
    assert download(client, 1, "train").status_code == 200
    assert download(client, 1, "val").status_code == 200

    # 从可信副本恢复 -> 原字节逐字节一致
    r = recover(client, 2, kind, "baseline")
    assert "两份索引均已恢复" in r.text
    assert download(client, 2, "train").content == train_before
    assert download(client, 2, "val").content == val_before
    # 版本号、分组、训练/验证归属不变
    assert "v2" in client.get("/versions/2").text
    assert client.get("/versions/2").text.count("<tr>") >= detail_before.count("<tr>")

    # 重启后仍判定为通过（状态来自磁盘 + 已记录摘要）
    fresh = plain_client(settings)
    assert download(fresh, 2, "train").status_code == 200
    assert "校验通过" in fresh.get("/versions/2").text


def test_uploading_wrong_bytes_is_rejected_and_keeps_downloads_paused(env):
    client, settings = env
    upload(client, BATCH1, "b1.csv")
    orig = idx_path(settings, 1, "train").read_bytes()
    idx_path(settings, 1, "train").write_bytes(b"photo_no,camera_no\nP1,C1\n")

    r = recover(client, 1, "train", "upload", b"photo_no,camera_no\nP9,C9\n")
    assert "不一致" in r.text
    assert idx_path(settings, 1, "train").read_bytes() != b"photo_no,camera_no\nP9,C9\n"
    assert download(client, 1, "train").status_code == 409
    assert download(client, 1, "val").status_code == 409

    # 上传正确的原字节 -> 恢复，且可信副本保持完好
    r2 = recover(client, 1, "train", "upload", orig)
    assert "两份索引均已恢复" in r2.text
    assert download(client, 1, "train").content == orig
    assert recovery_path(settings, 1, "train").read_bytes() == orig


def test_missing_file_restores_from_trusted_copy(env):
    client, settings = env
    upload(client, BATCH1, "b1.csv")
    orig = idx_path(settings, 1, "val").read_bytes()
    idx_path(settings, 1, "val").unlink()

    assert download(client, 1, "train").status_code == 409
    assert download(client, 1, "val").status_code == 409
    assert "文件缺失" in client.get("/versions/1").text

    r = recover(client, 1, "val", "baseline")
    assert "两份索引均已恢复" in r.text
    assert download(client, 1, "val").content == orig


def test_corrupt_trusted_copy_requires_original_upload(env):
    client, settings = env
    upload(client, BATCH1, "b1.csv")
    orig_train = idx_path(settings, 1, "train").read_bytes()
    # 当前公开文件被改坏，同时可信副本也丢了 -> 无法从副本恢复
    idx_path(settings, 1, "train").write_bytes(b"tampered")
    recovery_path(settings, 1, "train").unlink()

    r = recover(client, 1, "train", "baseline")
    assert "副本缺失或损坏" in r.text
    assert download(client, 1, "train").status_code == 409

    # 上传原字节可同时修复公开文件与可信副本
    r2 = recover(client, 1, "train", "upload", orig_train)
    assert "两份索引均已恢复" in r2.text
    assert download(client, 1, "train").content == orig_train
    assert recovery_path(settings, 1, "train").read_bytes() == orig_train


def test_damaged_trusted_copy_can_be_rebuilt_from_intact_release_file(env):
    client, settings = env
    upload(client, BATCH1, "b1.csv")
    orig = idx_path(settings, 1, "train").read_bytes()
    recovery_path(settings, 1, "train").write_bytes(b"copy-damaged")

    body = client.get("/versions/1").text
    assert "恢复依据损坏" in body
    assert download(client, 1, "train").status_code == 409
    assert download(client, 1, "val").status_code == 409

    r = recover(client, 1, "train", "repair")
    assert "两份索引均已恢复" in r.text
    assert recovery_path(settings, 1, "train").read_bytes() == orig
    assert idx_path(settings, 1, "train").read_bytes() == orig
    assert download(client, 1, "train").status_code == 200


# ---------------------------------------------------------------- 旧版建档

def _make_legacy(settings, *, drop=()):
    """构造一个"功能上线前发布"的版本：有版本目录与行，但无任何可信摘要。"""
    target = settings.versions_dir / "v9"
    target.mkdir(parents=True)
    train = b"photo_no,camera_no,captured_at,species,rel_path,group_id\nP1,C1,2026-03-01T08:00:00Z,boar,c/P1.jpg,P1\n"
    val = b"photo_no,camera_no,captured_at,species,rel_path,group_id\n"
    for kind, data in (("train", train), ("val", val)):
        if kind in drop:
            continue
        (target / f"{kind}_index.csv").write_bytes(data)
    conn = sqlite3.connect(settings.db_path)
    conn.execute(
        "INSERT INTO versions(id, created_at, photo_count, group_count, train_count, val_count,"
        " train_path, val_path, integrity_status) VALUES (9, ?, 1, 1, 1, 0, ?, ?, 'unbaselined')",
        ("2026-01-01T00:00:00Z", "versions/v9/train_index.csv", "versions/v9/val_index.csv"),
    )
    conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('active_version', '9')")
    conn.commit()
    conn.close()
    return target


def test_legacy_version_can_be_adopted_from_existing_files(env):
    client, settings = env
    upload(client, BATCH1, "b1.csv")  # 建库
    target = _make_legacy(settings)
    original = (target / "train_index.csv").read_bytes()

    body = client.get("/versions/9").text
    assert "旧版本" in body and "按现存内容建档" in body
    # 建档前仍可下载（保持原有行为），但会提示未建档
    assert download(client, 9, "train").status_code == 200

    r = client.post("/versions/9/adopt", follow_redirects=True)
    assert "已按现存内容建档" in r.text
    assert "校验通过" in client.get("/versions/9").text

    # 建档后任何改动都会被检出，并可从建档副本恢复
    (target / "train_index.csv").write_bytes(original.replace(b"boar", b"wolf"))
    assert download(client, 9, "train").status_code == 409
    assert download(client, 9, "val").status_code == 409
    r2 = recover(client, 9, "train", "baseline")
    assert "两份索引均已恢复" in r2.text
    assert (target / "train_index.csv").read_bytes() == original
    assert download(client, 9, "train").status_code == 200


def test_legacy_version_with_missing_file_cannot_be_adopted_or_recomputed(env):
    client, settings = env
    upload(client, BATCH1, "b1.csv")
    _make_legacy(settings, drop=("val",))
    conn = sqlite3.connect(settings.db_path)
    photos_before = conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0]
    versions_before = conn.execute("SELECT COUNT(*) FROM versions").fetchone()[0]
    conn.close()

    r = client.post("/versions/9/adopt", follow_redirects=True)
    assert "无法恢复" in r.text and "文件缺失" in r.text
    assert "不会根据当前照片表重算" in r.text
    # 不产生新版本、不改动照片数据、仍无摘要
    conn = sqlite3.connect(settings.db_path)
    assert conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0] == photos_before
    assert conn.execute("SELECT COUNT(*) FROM versions").fetchone()[0] == versions_before
    assert conn.execute("SELECT COUNT(*) FROM index_baselines WHERE version_id=9").fetchone()[0] == 0
    conn.close()
    # 两份下载都暂停（缺一份即无法校验）
    assert download(client, 9, "train").status_code == 409
    assert download(client, 9, "val").status_code == 409


# ---------------------------------------------------------------- 中断与重试

def test_interrupted_recovery_never_serves_half_recovered_index(env, monkeypatch):
    client, settings = env
    upload(client, BATCH1, "b1.csv")
    upload(client, BACKFILL, "b2.csv")
    train_orig = idx_path(settings, 2, "train").read_bytes()
    val_orig = idx_path(settings, 2, "val").read_bytes()
    idx_path(settings, 2, "train").write_bytes(b"train-tampered")
    idx_path(settings, 2, "val").write_bytes(b"val-tampered")

    # 让 val 的写入在 train 已恢复后失败（模拟恢复中途崩溃）
    real_atomic_write = integrity_mod.atomic_write

    def fail_on_val(path, data):
        if path == idx_path(settings, 2, "val"):
            raise OSError("磁盘写入失败（模拟中断）")
        return real_atomic_write(path, data)

    monkeypatch.setattr(integrity_mod, "atomic_write", fail_on_val)
    # 先恢复 train：单份成功，但 val 仍坏 -> 两份下载继续暂停（不提供半恢复）
    r = recover(client, 2, "train", "baseline")
    assert "另一份索引仍未通过校验" in r.text
    assert idx_path(settings, 2, "train").read_bytes() == train_orig
    assert download(client, 2, "train").status_code == 409
    assert download(client, 2, "val").status_code == 409
    assert "内容被改动" in client.get("/versions/2").text

    # 恢复 val 时写入中断：明确提示未提供半恢复的索引
    r_fail = recover(client, 2, "val", "baseline")
    assert "未提供半恢复" in r_fail.text
    assert idx_path(settings, 2, "val").read_bytes() == b"val-tampered"
    assert download(client, 2, "val").status_code == 409

    # 故障消除后重试可完成，两个文件都恢复原字节
    monkeypatch.setattr(integrity_mod, "atomic_write", real_atomic_write)
    r2 = recover(client, 2, "val", "baseline")
    assert "两份索引均已恢复" in r2.text
    assert download(client, 2, "train").content == train_orig
    assert download(client, 2, "val").content == val_orig

    # 重复恢复同一份是幂等的
    r3 = recover(client, 2, "val", "baseline")
    assert "两份索引均已恢复" in r3.text


def test_interrupted_publish_leaves_no_version_and_retry_completes(env, monkeypatch):
    client, settings = env
    upload(client, BATCH1, "b1.csv")
    v1_train = download(client, 1, "train").content

    def boom(*args, **kwargs):
        raise OSError("可信依据写入失败（模拟）")

    monkeypatch.setattr(main, "create_baseline", boom)
    r = upload(client, BACKFILL, "b2.csv")
    assert "发布失败" in r.text and "现役版本未变" in r.text
    assert not (settings.versions_dir / "v2").exists()
    assert not (settings.recovery_dir / "v2").exists()
    assert download(client, 1, "train").content == v1_train

    monkeypatch.undo()
    r2 = upload(client, BACKFILL, "b2.csv")
    assert "已发布版本 v2" in r2.text
    assert download(client, 2, "train").status_code == 200
    assert recovery_path(settings, 2, "train").is_file()


def test_interrupted_baseline_can_be_completed_by_retrying_publish(env):
    client, settings = env
    upload(client, BATCH1, "b1.csv")
    # 模拟"版本与文件已提交，但可信依据未写成"的极小窗口：删掉依据，模拟中断遗留。
    conn = sqlite3.connect(settings.db_path)
    conn.execute("DELETE FROM index_baselines WHERE version_id=1")
    conn.execute("UPDATE versions SET integrity_status='unbaselined' WHERE id=1")
    conn.commit()
    conn.close()
    import shutil

    shutil.rmtree(settings.recovery_dir / "v1")
    train_before = download(client, 1, "train").content

    body = client.get("/versions/1").text
    assert "尚无可信恢复依据" in body

    # 重投同一份 CSV 仍是 noop；版本与文件字节都不变
    r = upload(client, BATCH1, "b1-retry.csv")
    assert "未产生新版本" in r.text
    assert download(client, 1, "train").content == train_before

    # 页面提供"按现存文件补建"入口，重试可完成补建
    r2 = client.post("/versions/1/backfill", follow_redirects=True)
    assert "补建可信恢复依据" in r2.text
    conn = sqlite3.connect(settings.db_path)
    assert conn.execute("SELECT COUNT(*) FROM versions").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM index_baselines WHERE version_id=1").fetchone()[0] == 2
    conn.close()
    assert "校验通过" in client.get("/versions/1").text
    assert download(client, 1, "train").content == train_before


def test_recovery_interleaved_with_backfill_publish_stays_consistent(env):
    client, settings = env
    upload(client, BATCH1, "b1.csv")
    train_orig = idx_path(settings, 1, "train").read_bytes()
    idx_path(settings, 1, "train").write_bytes(b"tampered")

    backfill = make_csv([["P0100", "CAM20", "2026-03-05T10:00:00Z", "boar", "c20/P0100.jpg"]])
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_rec = pool.submit(recover, client, 1, "train", "baseline")
        f_pub = pool.submit(upload, client, backfill, "b3.csv")
        r_rec, r_pub = f_rec.result(), f_pub.result()

    assert "两份索引均已恢复" in r_rec.text
    assert "已发布版本 v2" in r_pub.text
    # 旧版本恢复为原字节，新版本完整且可下载；现役为新版本
    assert download(client, 1, "train").content == train_orig
    assert download(client, 1, "val").status_code == 200
    assert download(client, 2, "train").status_code == 200
    assert download(client, 2, "val").status_code == 200
    home = client.get("/").text
    assert "v2" in home
    conn = sqlite3.connect(settings.db_path)
    assert conn.execute("SELECT COUNT(*) FROM versions").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM index_baselines").fetchone()[0] == 4
    conn.close()


def test_other_versions_unaffected_when_one_version_is_corrupt(env):
    client, settings = env
    upload(client, BATCH1, "b1.csv")
    upload(client, BACKFILL, "b2.csv")
    upload(client, make_csv([["P0200", "CAM30", "2026-03-06T10:00:00Z", "deer", "c30/P0200.jpg"]]), "b3.csv")

    idx_path(settings, 2, "val").unlink()
    assert download(client, 1, "train").status_code == 200
    assert download(client, 1, "val").status_code == 200
    assert download(client, 3, "train").status_code == 200
    assert download(client, 3, "val").status_code == 200
    assert download(client, 2, "train").status_code == 409
    assert download(client, 2, "val").status_code == 409
    home = client.get("/").text
    assert home.count("下载已暂停") >= 1
    assert "v3" in home


def test_verify_endpoint_reports_status(env):
    client, settings = env
    upload(client, BATCH1, "b1.csv")
    r = client.post("/versions/1/verify", follow_redirects=True)
    assert "校验通过" in r.text
    idx_path(settings, 1, "val").write_bytes(b"x")
    r2 = client.post("/versions/1/verify", follow_redirects=True)
    assert "校验未通过" in r2.text and "内容被改动" in r2.text
