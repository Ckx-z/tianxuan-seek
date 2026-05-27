"""v3 单体配对过滤 — 单体池构建 + 硬约束过滤。

硬约束:
  - 排金属单体
  - C1 排除 (单官能团无法形成二维网络)
  - C4+C4 排除 (不可控交联)
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Optional

from rdkit import Chem, RDLogger

from src.chemistry.monomer import has_metal_smiles, has_metal_name

RDLogger.logger().setLevel(RDLogger.ERROR)


@dataclass
class MonomerInfo:
    smiles: str
    name: str
    source: str
    monomer_type: str
    n_fg: int = 0
    has_fluorine: bool = False
    mol: Optional[Chem.Mol] = None

    def __post_init__(self):
        if self.mol is None and self.smiles:
            mol = Chem.MolFromSmiles(self.smiles)
            if mol is None:
                mol = Chem.MolFromSmiles(self.smiles, sanitize=False)
            self.mol = mol


def count_aldehyde_groups(mol: Chem.Mol) -> int:
    if mol is None:
        return 0
    count = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6:
            continue
        for nb in atom.GetNeighbors():
            if nb.GetAtomicNum() == 8:
                bond = mol.GetBondBetweenAtoms(atom.GetIdx(), nb.GetIdx())
                if bond and bond.GetBondType() == Chem.BondType.DOUBLE and nb.GetDegree() == 1:
                    count += 1
                    break
    return count


def count_amine_groups(mol: Chem.Mol) -> int:
    if mol is None:
        return 0
    count = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 7:
            continue
        heavy = [n for n in atom.GetNeighbors() if n.GetAtomicNum() != 1]
        if len(heavy) != 1:
            continue
        if sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 8) >= 2:
            continue
        count += 1
    return count


def load_monomer_pool(train_csv: str,
                      commercial_csv: str | None = None,
                      min_freq: int = 2
                      ) -> tuple[list[MonomerInfo], list[MonomerInfo]]:
    from collections import Counter

    ald_dict: dict[str, MonomerInfo] = {}
    amine_dict: dict[str, MonomerInfo] = {}

    with open(train_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    ald_freq = Counter(r["aldehyde_smiles"].strip() for r in rows)
    amine_freq = Counter(r["amine_smiles"].strip() for r in rows)

    for r in rows:
        for smi_key, name_key, mtype, pool, freq in [
            ("aldehyde_smiles", "aldehyde_name", "aldehyde", ald_dict, ald_freq),
            ("amine_smiles", "amine_name", "amine", amine_dict, amine_freq),
        ]:
            smi = r[smi_key].strip()
            if not smi or smi in pool:
                continue
            if freq[smi] < min_freq:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                mol = Chem.MolFromSmiles(smi, sanitize=False)
            if mol is None:
                continue
            name = r.get(name_key, "") or smi
            if has_metal_smiles(smi) or has_metal_name(name):
                continue
            n_fg = count_aldehyde_groups(mol) if mtype == "aldehyde" else count_amine_groups(mol)
            if n_fg < 2:
                continue
            has_f = any(a.GetAtomicNum() == 9 for a in mol.GetAtoms())
            pool[smi] = MonomerInfo(
                smiles=smi, name=name, source="training",
                monomer_type=mtype, n_fg=n_fg, has_fluorine=has_f, mol=mol,
            )

    if commercial_csv and os.path.exists(commercial_csv):
        with open(commercial_csv, "r", encoding="utf-8") as f:
            comm_rows = list(csv.DictReader(f))
        for r in comm_rows:
            smi = (r.get("smiles") or "").strip()
            if not smi:
                continue
            mtype = (r.get("monomer_type") or "").strip().lower()
            name = r.get("name", "")
            target = ald_dict if mtype == "aldehyde" else amine_dict
            if smi in target:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                mol = Chem.MolFromSmiles(smi, sanitize=False)
            if mol is None:
                continue
            if has_metal_smiles(smi) or has_metal_name(name):
                continue
            n_fg = count_aldehyde_groups(mol) if mtype == "aldehyde" else count_amine_groups(mol)
            has_f = any(a.GetAtomicNum() == 9 for a in mol.GetAtoms())
            target[smi] = MonomerInfo(
                smiles=smi, name=name, source="commercial",
                monomer_type=mtype, n_fg=n_fg, has_fluorine=has_f, mol=mol,
            )

    return list(ald_dict.values()), list(amine_dict.values())


def check_hard_constraints(ald: MonomerInfo, amine: MonomerInfo) -> tuple[bool, str]:
    if ald.n_fg < 2:
        return False, f"醛官能团={ald.n_fg}"
    if amine.n_fg < 2:
        return False, f"胺官能团={amine.n_fg}"
    if ald.n_fg >= 4 and amine.n_fg >= 4:
        return False, "C4+C4"

    def topo(n):
        return f"C{min(n, 4)}"

    return True, f"{topo(ald.n_fg)}+{topo(amine.n_fg)}"


def generate_pairs(aldehydes: list[MonomerInfo], amines: list[MonomerInfo],
                   train_pairs: set[tuple[str, str]] | None = None
                   ) -> list[dict]:
    train_pairs = train_pairs or set()
    results = []
    for ald in aldehydes:
        for amine in amines:
            passed, reason = check_hard_constraints(ald, amine)
            results.append({
                "ald": ald, "amine": amine,
                "hard_pass": passed,
                "topology": reason if passed else f"EXCLUDED:{reason}",
                "in_training_set": (ald.smiles, amine.smiles) in train_pairs,
            })
    return results
