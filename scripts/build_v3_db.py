"""v3 数据库构建脚本 — 合并 structured_v2 (数据库1) + structured_v3 (数据库2) → fluorofilm_v3.db。

两库合并后去重（DOI + 标题模糊匹配），构建统一的 papers + experiments 表。

Usage:
  python scripts/build_v3_db.py                    # 全量构建
  python scripts/build_v3_db.py --dry-run          # 预检，不写入
  python scripts/build_v3_db.py --db data/fluorofilm_v3.db  # 自定义路径
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Optional

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.extraction.v2_validator import validate_document, filter_valid_groups
from src.utils.logger import setup_logger

logger = setup_logger("build_v3_db")

STRUCTURED_DIRS = ["data/structured_v2", "data/structured_v3"]
DEFAULT_DB = "data/fluorofilm_v3.db"

# ── 表结构 ──────────────────────────────────────────────────

PAPERS_DDL = """
CREATE TABLE IF NOT EXISTS papers (
    paper_id        TEXT PRIMARY KEY,
    title           TEXT,
    doi             TEXT,
    journal         TEXT,
    has_si          INTEGER,
    total_experiments INTEGER,
    chemistry_type  TEXT,
    innovation      TEXT,
    experimental_logic TEXT,
    key_conclusion  TEXT,
    limitations     TEXT,
    source_db       TEXT,
    created_at      TEXT
)
"""

EXPERIMENTS_DDL = """
CREATE TABLE IF NOT EXISTS experiments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id        TEXT NOT NULL,
    group_id        INTEGER NOT NULL,
    group_name      TEXT,
    design_rationale TEXT,
    comparison      TEXT,
    notes           TEXT,

    -- monomers
    aldehyde_name   TEXT,
    aldehyde_smiles TEXT,
    amine_name      TEXT,
    amine_smiles    TEXT,
    stoichiometry   TEXT,

    -- conditions
    solvent             TEXT,
    monomer_concentration TEXT,
    temperature         TEXT,
    duration            TEXT,
    catalyst            TEXT,
    catalyst_amount     TEXT,
    synthesis_route     TEXT,
    interface_type      TEXT,
    atmosphere          TEXT,
    additives           TEXT,
    post_treatment      TEXT,
    substrate           TEXT,

    -- film_result
    is_film         INTEGER,
    film_type       TEXT,
    film_description TEXT,
    film_quality    TEXT,

    -- evidence
    from_text       TEXT,
    from_figure     TEXT,
    xrd_peaks       TEXT,
    afm_rms         TEXT,
    bet_surface_area TEXT,
    youngs_modulus  TEXT,
    ftir_cn_peak    TEXT,
    crystallinity   TEXT,
    confidence      TEXT,
    confidence_note TEXT,

    -- fluorine
    has_fluorine_monomer INTEGER,
    fluorine_content TEXT,
    fluorine_effect  TEXT,

    -- heterocycle
    has_n_heterocycle INTEGER,
    heterocycle_type TEXT,

    -- characterization
    char_methods    TEXT,
    char_key_findings TEXT,

    -- metadata
    is_valid        INTEGER DEFAULT 1,
    FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
)
"""

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_exp_paper_id ON experiments(paper_id)",
    "CREATE INDEX IF NOT EXISTS idx_exp_is_film ON experiments(is_film)",
    "CREATE INDEX IF NOT EXISTS idx_exp_confidence ON experiments(confidence)",
    "CREATE INDEX IF NOT EXISTS idx_exp_synthesis_route ON experiments(synthesis_route)",
    "CREATE INDEX IF NOT EXISTS idx_exp_has_fluorine ON experiments(has_fluorine_monomer)",
    "CREATE INDEX IF NOT EXISTS idx_exp_has_n_heterocycle ON experiments(has_n_heterocycle)",
    "CREATE INDEX IF NOT EXISTS idx_exp_is_valid ON experiments(is_valid)",
]


def _safe_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _safe_bool(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, str):
        return 1 if value.lower() in ("true", "yes", "1") else 0
    return 1 if value else 0


def _normalize_title(title: str) -> str:
    """标题标准化用于模糊匹配去重。"""
    t = title.lower().strip()
    t = re.sub(r'[^a-z0-9\s]', '', t)
    t = re.sub(r'\s+', ' ', t)
    return t.strip()


def _load_yaml_files(dirs: list[str]) -> list[tuple[str, str, dict]]:
    """加载所有 YAML 文件，返回 [(source_dir, filename, data), ...]"""
    results = []
    for d in dirs:
        if not os.path.isdir(d):
            logger.warning(f"目录不存在: {d}")
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith(".yaml"):
                continue
            path = os.path.join(d, f)
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = yaml.safe_load(fh)
                if data and isinstance(data, dict):
                    results.append((d, f, data))
            except Exception as e:
                logger.warning(f"YAML 加载失败 [{f}]: {e}")
    return results


def _deduplicate_papers(
    papers: list[tuple[str, str, dict]],
) -> tuple[list[tuple[str, str, dict]], list[dict]]:
    """DOI + 标题模糊匹配去重。

    保留优先: structured_v2 (数据库1) > structured_v3 (数据库2)
    返回: (去重后的 papers, 被剔除的 duplicates 列表)
    """
    seen_doi: dict[str, str] = {}  # doi → paper_id
    seen_titles: list[tuple[str, str, str]] = []  # [(normalized_title, paper_id, source)]
    kept = []
    duplicates = []

    # 按优先级排序: v2 优先
    prioritized = sorted(papers, key=lambda x: 0 if "v2" in x[0] else 1)

    for src_dir, fname, data in prioritized:
        pid = data.get("paper_id", fname.replace(".yaml", ""))
        doi = (data.get("doi") or "").strip().lower()
        title = (data.get("title") or "").strip()

        # DOI 去重
        if doi and doi in seen_doi:
            dup_info = {"paper_id": pid, "doi": doi, "title": title,
                        "source": src_dir, "dup_of": seen_doi[doi]}
            duplicates.append(dup_info)
            continue

        # 标题模糊匹配
        norm_title = _normalize_title(title)
        is_dup = False
        for nt, existing_pid, existing_src in seen_titles:
            if len(norm_title) < 20 or len(nt) < 20:
                continue
            sim = SequenceMatcher(None, norm_title, nt).ratio()
            if sim > 0.85:
                dup_info = {"paper_id": pid, "doi": doi, "title": title,
                            "source": src_dir, "dup_of": existing_pid,
                            "similarity": sim}
                duplicates.append(dup_info)
                is_dup = True
                break
        if is_dup:
            continue

        if doi:
            seen_doi[doi] = pid
        seen_titles.append((norm_title, pid, src_dir))
        kept.append((src_dir, fname, data))

    return kept, duplicates


def build_db(db_path: str, dry_run: bool = False) -> dict:
    """从两库 YAML 文件合并构建 v3 数据库。"""
    all_papers = _load_yaml_files(STRUCTURED_DIRS)
    logger.info(f"加载 {len(all_papers)} 篇文献 (来自 {STRUCTURED_DIRS})")

    kept, duplicates = _deduplicate_papers(all_papers)
    logger.info(f"去重: {len(kept)} 保留, {len(duplicates)} 剔除")

    # 按来源统计
    src_counts = defaultdict(int)
    for src_dir, _, _ in kept:
        src_counts[src_dir] += 1
    logger.info(f"来源分布: {dict(src_counts)}")

    if dry_run:
        exp_total = 0
        film_total = 0
        for src_dir, fname, data in kept:
            filled, _ = validate_document(data)
            filled = filter_valid_groups(filled)
            exps = filled.get("experiments", [])
            exp_total += len(exps)
            film_total += sum(
                1 for e in exps
                if e.get("film_result", {}).get("is_film") in (True, 1, "1")
            )
        logger.info(f"预检: {len(kept)} 篇, {exp_total} 实验组, {film_total} 成膜, {len(duplicates)} 重复")
        return {
            "paper_count": len(kept), "exp_count": exp_total,
            "film_count": film_total, "duplicates": len(duplicates),
            "source_breakdown": dict(src_counts),
        }

    # ── 正式构建 ──
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute(PAPERS_DDL)
    conn.execute(EXPERIMENTS_DDL)
    for idx_sql in INDEXES:
        conn.execute(idx_sql)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    paper_count = 0
    exp_count = 0
    film_count = 0
    error_count = 0

    for src_dir, fname, data in kept:
        try:
            filled, errs = validate_document(data)
            filled = filter_valid_groups(filled)
            pid = filled.get("paper_id", fname.replace(".yaml", ""))
            summary = filled.get("paper_summary", {}) or {}

            # source_db 标记
            source_label = "db1" if "v2" in src_dir else "db2"

            conn.execute(
                """INSERT OR REPLACE INTO papers
                   (paper_id, title, doi, journal, has_si, total_experiments,
                    chemistry_type, innovation, experimental_logic, key_conclusion,
                    limitations, source_db, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    _safe_str(pid),
                    _safe_str(filled.get("title")),
                    _safe_str(filled.get("doi")),
                    _safe_str(filled.get("journal")),
                    _safe_bool(filled.get("has_si")),
                    len(filled.get("experiments", [])),
                    _safe_str(summary.get("chemistry_type")),
                    _safe_str(summary.get("innovation")),
                    _safe_str(summary.get("experimental_logic")),
                    _safe_str(summary.get("key_conclusion")),
                    _safe_str(summary.get("limitations")),
                    source_label,
                    now,
                ),
            )
            paper_count += 1

            for exp in filled.get("experiments", []):
                mono = exp.get("monomers", {}) or {}
                cond = exp.get("conditions", {}) or {}
                film = exp.get("film_result", {}) or {}
                evid = exp.get("evidence", {}) or {}
                fluo = exp.get("fluorine", {}) or {}
                hetero = exp.get("heterocycle", {}) or {}
                chara = exp.get("characterization", {}) or {}

                is_film = film.get("is_film")
                is_valid = not exp.get("_invalid", False)

                conn.execute(
                    """INSERT INTO experiments
                       (paper_id, group_id, group_name, design_rationale, comparison, notes,
                        aldehyde_name, aldehyde_smiles, amine_name, amine_smiles, stoichiometry,
                        solvent, monomer_concentration, temperature, duration,
                        catalyst, catalyst_amount, synthesis_route, interface_type,
                        atmosphere, additives, post_treatment, substrate,
                        is_film, film_type, film_description, film_quality,
                        from_text, from_figure, xrd_peaks, afm_rms, bet_surface_area,
                        youngs_modulus, ftir_cn_peak, crystallinity, confidence, confidence_note,
                        has_fluorine_monomer, fluorine_content, fluorine_effect,
                        has_n_heterocycle, heterocycle_type,
                        char_methods, char_key_findings,
                        is_valid)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                               ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                               ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                               ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        _safe_str(pid),
                        exp.get("group_id"),
                        _safe_str(exp.get("group_name")),
                        _safe_str(exp.get("design_rationale")),
                        _safe_str(exp.get("comparison")),
                        _safe_str(exp.get("notes")),
                        _safe_str(mono.get("aldehyde_name")),
                        _safe_str(mono.get("aldehyde_smiles")),
                        _safe_str(mono.get("amine_name")),
                        _safe_str(mono.get("amine_smiles")),
                        _safe_str(mono.get("stoichiometry")),
                        _safe_str(cond.get("solvent")),
                        _safe_str(cond.get("monomer_concentration")),
                        _safe_str(cond.get("temperature")),
                        _safe_str(cond.get("duration")),
                        _safe_str(cond.get("catalyst")),
                        _safe_str(cond.get("catalyst_amount")),
                        _safe_str(cond.get("synthesis_route")),
                        _safe_str(cond.get("interface_type")),
                        _safe_str(cond.get("atmosphere")),
                        _safe_str(cond.get("additives")),
                        _safe_str(cond.get("post_treatment")),
                        _safe_str(cond.get("substrate")),
                        _safe_bool(is_film),
                        _safe_str(film.get("film_type")),
                        _safe_str(film.get("film_description")),
                        _safe_str(film.get("film_quality")),
                        _safe_str(evid.get("from_text")),
                        _safe_str(evid.get("from_figure")),
                        _safe_str(evid.get("xrd_peaks")),
                        _safe_str(evid.get("afm_rms")),
                        _safe_str(evid.get("bet_surface_area")),
                        _safe_str(evid.get("youngs_modulus")),
                        _safe_str(evid.get("ftir_cn_peak")),
                        _safe_str(evid.get("crystallinity")),
                        _safe_str(evid.get("confidence")),
                        _safe_str(evid.get("confidence_note")),
                        _safe_bool(fluo.get("has_fluorine_monomer")),
                        _safe_str(fluo.get("fluorine_content")),
                        _safe_str(fluo.get("fluorine_effect")),
                        _safe_bool(hetero.get("has_n_heterocycle")),
                        _safe_str(hetero.get("heterocycle_type")),
                        _safe_str(chara.get("methods")),
                        _safe_str(chara.get("key_findings")),
                        1 if is_valid else 0,
                    ),
                )
                exp_count += 1
                if is_film in (True, 1, "1", "true", "True"):
                    film_count += 1

            if errs:
                error_count += 1

        except Exception as e:
            logger.warning(f"入库失败 [{fname}]: {e}")
            error_count += 1

    conn.commit()

    paper_total = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    exp_total = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
    exp_valid = conn.execute("SELECT COUNT(*) FROM experiments WHERE is_valid=1").fetchone()[0]
    exp_film = conn.execute("SELECT COUNT(*) FROM experiments WHERE is_film=1").fetchone()[0]

    # 按来源统计
    db1_papers = conn.execute("SELECT COUNT(*) FROM papers WHERE source_db='db1'").fetchone()[0]
    db2_papers = conn.execute("SELECT COUNT(*) FROM papers WHERE source_db='db2'").fetchone()[0]
    db1_exps = conn.execute(
        "SELECT COUNT(*) FROM experiments e JOIN papers p ON e.paper_id=p.paper_id WHERE p.source_db='db1'"
    ).fetchone()[0]
    db2_exps = conn.execute(
        "SELECT COUNT(*) FROM experiments e JOIN papers p ON e.paper_id=p.paper_id WHERE p.source_db='db2'"
    ).fetchone()[0]

    conn.close()

    print(f"\n{'='*60}")
    print(f"  v3 数据库构建完成 (两库合并)")
    print(f"  数据库: {db_path}")
    print(f"  文献: {paper_total} 篇 (数据库1: {db1_papers}, 数据库2: {db2_papers})")
    print(f"  实验组: {exp_total} (数据库1: {db1_exps}, 数据库2: {db2_exps})")
    print(f"  有效: {exp_valid}, 成膜: {exp_film}")
    print(f"  去重剔除: {len(duplicates)} 篇")
    print(f"  校验错误: {error_count}")
    print(f"{'='*60}")

    return {
        "paper_count": paper_total, "exp_count": exp_total,
        "valid_exp_count": exp_valid, "film_count": exp_film,
        "duplicates": len(duplicates), "errors": error_count,
        "db1_papers": db1_papers, "db2_papers": db2_papers,
        "db1_exps": db1_exps, "db2_exps": db2_exps,
    }


def main():
    parser = argparse.ArgumentParser(description="v3 数据库构建 (两库合并)")
    parser.add_argument("--db", type=str, default=DEFAULT_DB, help="数据库路径")
    parser.add_argument("--dry-run", action="store_true", help="预检模式")
    args = parser.parse_args()

    build_db(args.db, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
