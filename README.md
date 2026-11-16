# 野生动物红外相机照片索引服务

面向物种识别模型训练的数据整理工具：研究员分批上传红外相机照片 CSV，服务把
**同一次出没**的照片传递归组，按组稳定划分训练/验证索引，每批新增照片发布一个
**不可变版本**（两份可下载索引 + 版本号）。

技术栈：FastAPI + SQLite（WAL），单容器即可运行。

## 快速开始

```bash
docker compose up --build
# 打开 http://localhost:8000
```

数据（SQLite 库 + 各版本索引文件）持久化在命名卷 `wildlife-data` 中，容器重建不丢。

## 浏览器演示：跨批补传样例

首页可直接下载 `samples/` 下的四个样例 CSV（仓库目录 `samples/` 中也有），按顺序操作：

| 步骤 | 上传文件 | 预期结果 |
|---|---|---|
| 1 | `batch1_base.csv` | 发布 **v1**：6 张 / 4 组。CAM01 两段出没间隔 70 秒 > 60，分为两组 |
| 2 | `batch2_backfill.csv` | 发布 **v2**：补传的 P0007（08:01:05）距前后两组各 35 秒，把 CAM01 旧两组**合并**；P0008 加入 CAM02 旧组；P0009 新相机新组。全库重新归组 |
| 3 | `batch3_duplicate.csv` | 与 batch2 完全同号同内容 → **不产生新版本**（noop） |
| 4 | `batch4_conflict.csv` | P0001 物种被改动 → **整批拒绝**（含其中的新照片 P0011），现役版本保持 v2 |

任意时刻打开 v1 的下载链接，内容保持最初发布时的原样（版本不可变）。
首页「版本历史」可进入每个版本查看分组明细，「批次记录」查看每批的处理结果与拒绝原因。

## CSV 格式

必需列（顺序不限，表头名需一致，UTF-8 编码）：

```csv
photo_no,camera_no,captured_at,species,rel_path
P0002,CAM01,2026-03-01T08:00:00Z,wild_boar,cam01/20260301/P0002.jpg
```

- `captured_at`：UTC 拍摄时间，ISO8601（`2026-03-01T08:00:00Z`；带时区偏移会换算为 UTC，无时区按 UTC 处理）
- 任何一行校验失败（缺列、时间为空/无法解析等）→ **整批拒绝**，库内不产生任何变化

## 归组与划分规则

1. **归组**：同一 `camera_no` 内按 `(拍摄时间, 照片号)` 排序，相邻间隔 ≤ 60 秒
   （`GAP_SECONDS` 可调）即同组，分组可传递（链式桥接的照片同属一组）。
2. **组标识**：`group_id` = 组内最早照片的照片号。补传合并两组时沿用更早组的标识；
   若补传照片比组内最早照片还早，组标识随之改变。
3. **划分**：按组整体划分（同一出没事件不会泄漏到训练/验证两侧），
   `sha256(SPLIT_SALT | group_id) % 100 < VAL_PERCENT` → 验证集。
   完全确定：同样的照片集合无论分几批、以什么顺序入库，最终划分一致。

## 版本与一致性设计

- **发布原子性**：照片入库、索引文件落盘（先写 `.tmp-*` 目录再 `rename` 原子生效）、
  `versions` 行、`active_version` 指针在同一个 `BEGIN IMMEDIATE` 事务中提交。
  任一步失败整体回滚并清理半成品目录，**现役版本不变**；失败批次留有审计记录。
- **并发**：SQLite 写事务把交错上传（含多进程）串行化，后提交者基于前者已发布的
  状态继续；读者（页面/下载）走 WAL 快照，不被写入阻塞。页面展示的现役版本与两份
  下载始终对应同一完整版本。
- **不可变版本**：`data/versions/v{N}/` 一旦发布永不修改，旧链接永远下载原内容。
- **幂等**：同号同内容重投 → 批次记 `noop`，不产生新版本；
  同号异内容 → 整批 `rejected`。

## 本地开发与测试

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest tests/ -q          # 15 个测试：归组逻辑 + 端到端（含并发、发布失败回滚）
DATA_DIR=./data SAMPLES_DIR=./samples uvicorn app.main:app --reload
```

## 配置项（环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `DATA_DIR` | `./data`（容器内 `/data`） | SQLite 与版本索引存放目录 |
| `SAMPLES_DIR` | `./samples` | 样例 CSV 目录（存在时首页显示演示入口） |
| `VAL_PERCENT` | `20` | 验证集目标比例（按组分桶） |
| `GAP_SECONDS` | `60` | 同相机相邻照片归组阈值（秒） |
| `SPLIT_SALT` | `wildlife-trap-v1` | 划分哈希盐；改盐整体重排训练/验证划分 |

## 项目结构

```
app/
  main.py        # FastAPI 应用：上传、版本发布、页面、下载（含事务与落盘逻辑）
  grouping.py    # 归组与稳定划分（纯函数，可单测）
samples/         # 跨批补传演示的四个样例 CSV
tests/           # 单元测试 + 端到端测试
Dockerfile
docker-compose.yml
```
