import sqlite3
import csv
from datetime import datetime
from typing import List, Dict, Optional

# SQLite 表结构：21 个提取字段 + 元数据
FIELDS = [
    "literature_id", "journal", "system",
    "conclusion_1", "conclusion_2", "conclusion_3",
    "methods", "reagent", "catalyst", "solvent",
    "innovation", "film_crystallinity_fluorine", "reaction_temperature",
    "schiff_base_kinetics", "fluorine_effects", "adsorption_mechanism",
    "computational_methods", "interface_type", "annealing_conditions",
    "synthesis_route", "fluorine_monomer", "synthesis_mode",
]


def init_db(db_path: str) -> sqlite3.Connection:
    """创建文献数据库表，返回连接"""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    # 所有提取字段用 TEXT，literature_id 唯一主键
    columns = ["literature_id TEXT PRIMARY KEY", "created_at TEXT"]
    for field in FIELDS[1:]:  # 跳过 literature_id，已作为主键
        columns.append(f"{field} TEXT")
    ddl = f"CREATE TABLE IF NOT EXISTS literature ({', '.join(columns)})"
    conn.execute(ddl)
    conn.commit()
    return conn


def insert_record(conn: sqlite3.Connection, literature_id: str,
                  data: Dict[str, Optional[str]]):
    """插入或更新一条文献记录"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    columns = ["literature_id", "created_at"] + FIELDS[1:]
    placeholders = ", ".join(["?"] * len(columns))
    values = [literature_id, now]
    for field in FIELDS[1:]:
        values.append(data.get(field))
    sql = f"INSERT OR REPLACE INTO literature ({', '.join(columns)}) VALUES ({placeholders})"
    conn.execute(sql, values)
    conn.commit()


def get_all_records(conn: sqlite3.Connection) -> List[Dict]:
    """查询全部记录"""
    rows = conn.execute(
        "SELECT literature_id, created_at, " +
        ", ".join(FIELDS[1:]) + " FROM literature ORDER BY created_at"
    ).fetchall()
    keys = ["literature_id", "created_at"] + FIELDS[1:]
    return [dict(zip(keys, row)) for row in rows]


def get_record_count(conn: sqlite3.Connection) -> int:
    """返回记录总数"""
    return conn.execute("SELECT COUNT(*) FROM literature").fetchone()[0]


def export_to_csv(conn: sqlite3.Connection, csv_path: str):
    """导出全部记录为 CSV（供 pandas/ML 使用）"""
    records = get_all_records(conn)
    if not records:
        return
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
