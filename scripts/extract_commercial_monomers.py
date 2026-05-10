"""从商业单体 PDF 提取 CAS号/名称/化学式 → SMILES.

使用 MiMo Omni 多模态模型识别扫描页面中的化学结构信息。
"""
import argparse
import base64
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import setup_logger

logger = setup_logger("commercial")

SYSTEM_PROMPT = """你是一个化学信息提取助手。从这张产品目录页面图片中，提取所有COF/MOF配体单体。

对每个产品，提取以下字段：
1. 产品编号 (如 YSWK-A01001)
2. 中文名称
3. CAS号
4. 分子式 (如 C12H6O6)

请以 JSON 数组格式输出，每个产品一个对象：
```json
[{"id": "编号", "name": "中文名", "cas": "CAS号", "formula": "分子式"}]
```

规则：
- 只提取明确的化学配体/单体产品，忽略催化剂、溶剂、通用试剂
- 如果某个字段无法识别，填 null
- 中文名称使用完整的化学命名
- CAS号格式为 XXX-XX-X
- 分子式只包含元素和数字，如 C6H12O6
- 忽略价格、规格、包装等信息"""


def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_from_page(client, model: str, img_path: str, retry: int = 3) -> list[dict]:
    """用 MiMo Omni 从单页图片提取产品信息。"""
    b64 = encode_image(img_path)

    for attempt in range(1, retry + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64}"},
                            },
                            {
                                "type": "text",
                                "text": "请从这张产品目录页面中提取所有COF/MOF配体单体信息（JSON格式）。",
                            },
                        ],
                    },
                ],
                temperature=0.1,
                max_tokens=4096,
            )
            content = resp.choices[0].message.content
            return _parse_response(content)
        except Exception as e:
            logger.warning(f"  MiMo API 失败 (attempt {attempt}/{retry}): {e}")
            if attempt < retry:
                time.sleep(5)
    logger.error(f"  MiMo API 重试 {retry} 次后仍失败")
    return []


def _parse_response(content: str) -> list[dict]:
    """从 MiMo 回复中解析 JSON 数组。"""
    # 提取 JSON 数组
    match = re.search(r"\[.*\]", content, re.DOTALL)
    if not match:
        logger.warning(f"  未找到 JSON 数组: {content[:200]}")
        return []
    try:
        data = json.loads(match.group())
        return [d for d in data if isinstance(d, dict)]
    except json.JSONDecodeError:
        # 尝试修复常见问题
        try:
            cleaned = match.group().replace("\n", " ").replace("\r", "")
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning(f"  JSON 解析失败: {match.group()[:200]}")
            return []


def _pubchem_smiles(identifier: str) -> str | None:
    """通过 PubChem PUG REST 查询 SMILES。同时尝试 CanonicalSMILES 和 IsomericSMILES。"""
    import urllib.request
    import urllib.error

    for prop in ("CanonicalSMILES", "IsomericSMILES"):
        try:
            url = (
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
                f"{urllib.request.quote(identifier)}/property/{prop}/JSON"
            )
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "FluoroFilm/1.0")
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
            props = data.get("PropertyTable", {}).get("Properties", [])
            if props:
                for field in (prop, "ConnectivitySMILES", "CanonicalSMILES", "IsomericSMILES"):
                    smi = props[0].get(field)
                    if smi:
                        return smi
        except Exception:
            continue
    return None


def cas_to_smiles(cas: str) -> str | None:
    if not cas or cas == "null":
        return None
    return _pubchem_smiles(cas)


def name_to_smiles(name: str) -> str | None:
    if not name or name == "null":
        return None
    # 移除逗号后的额外描述 (如 "1,4-苯二甲醛,2,3,5,6-四氟-")
    clean = name.split(",")[0].strip() if "," in name else name
    return _pubchem_smiles(clean)


def formula_to_smiles(formula: str) -> str | None:
    if not formula or formula == "null":
        return None
    return _pubchem_smiles(formula)


def resolve_smiles(entry: dict, cache: dict) -> str | None:
    """多级 SMILES 解析: CAS > 名称 > 分子式。先查缓存。"""
    cas = entry.get("cas", "")
    name = entry.get("name", "")
    formula = entry.get("formula", "")

    # 缓存查询
    cache_key = cas if cas else name
    if cache_key and cache_key in cache:
        return cache[cache_key]

    # 1. 尝试 CAS
    smi = cas_to_smiles(cas)
    if smi:
        cache[cache_key] = smi
        return smi

    # 2. 尝试名称
    smi = name_to_smiles(name)
    if smi:
        cache[cache_key] = smi
        return smi

    # 3. 尝试分子式
    smi = formula_to_smiles(formula)
    if smi:
        cache[cache_key] = smi
        return smi

    return None


def filter_valid_monomers(entries: list[dict]) -> list[dict]:
    """过滤掉非单体产品 (催化剂、溶剂、试剂等)。"""
    import re

    # 非单体关键词
    SKIP_PATTERNS = [
        r"催化剂", r"catalyst", r"溶剂", r"solvent",
        r"干燥剂", r"分子筛", r"硅胶", r"氧化铝",
        r"缓冲液", r"缓冲溶液", r"标准溶液",
        r"柱层析", r"色谱", r"试纸", r"指示剂",
        r"硅油", r"真空脂", r"密封",
        r"注射器", r"针头", r"滤膜", r"滤纸",
        r"手套", r"口罩", r"护目镜",
        r"电极", r"电池", r"电解",
        r"氘代", r"标准品", r"对照品",
        r"试剂盒", r"检测试剂",
        r"纳米粒子", r"量子点", r"上转换",
    ]

    valid = []
    for e in entries:
        name = e.get("name", "")
        if not name or name == "null":
            # 如果连名称都没有，保留（后续通过 CAS 查 SMILES）
            if e.get("cas") and e.get("cas") != "null":
                valid.append(e)
            continue

        skip = False
        for pat in SKIP_PATTERNS:
            if re.search(pat, name, re.IGNORECASE):
                skip = True
                break
        if not skip:
            valid.append(e)
    return valid


def main():
    parser = argparse.ArgumentParser(description="从商业单体 PDF 提取 CAS/名称/分子式 → SMILES")
    parser.add_argument("--pdf", required=True, help="商业单体 PDF 路径")
    parser.add_argument("--output", default="data/processed/commercial_monomers.csv")
    parser.add_argument("--page-dir", default="data/tmp/pages")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--end-page", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=5, help="每批页数，批间休息")
    parser.add_argument("--cache", default="data/processed/monomer_smiles_cache.json")
    args = parser.parse_args()

    from openai import OpenAI
    import fitz
    import pandas as pd
    from rdkit import Chem

    # 初始化 MiMo Omni
    api_key = "tp-cco1ldct7lxvg5d9navn8t8j8xcnu9wgih6jxe2i16sr9844"
    base_url = "https://token-plan-cn.xiaomimimo.com/v1"
    model = "mimo-v2-omni"

    client = OpenAI(api_key=api_key, base_url=base_url)

    # 打开 PDF
    doc = fitz.open(args.pdf)
    total_pages = doc.page_count
    end = args.end_page if args.end_page > 0 else total_pages
    end = min(end, total_pages)
    start = max(1, args.start_page) - 1
    logger.info(f"PDF: {total_pages} 页, 处理 {start+1}-{end}")

    # 创建页面图片目录
    os.makedirs(args.page_dir, exist_ok=True)

    # 加载 SMILES 缓存
    cache = {}
    if os.path.exists(args.cache):
        with open(args.cache, "r", encoding="utf-8") as f:
            cache = json.load(f)
    logger.info(f"SMILES 缓存: {len(cache)} 条")

    all_entries = []

    for pno in range(start, end):
        page_num = pno + 1
        img_path = os.path.join(args.page_dir, f"p{pno:03d}.png")
        logger.info(f"  [{page_num}/{end}] 提取页面...")

        # 渲染页面为图片
        if not os.path.exists(img_path):
            page = doc[pno]
            mat = page.get_pixmap(dpi=args.dpi)
            mat.save(img_path)

        # 发送到 MiMo Omni
        entries = extract_from_page(client, model, img_path)
        if entries:
            all_entries.extend(entries)
            logger.info(f"    提取 {len(entries)} 个产品")

        # 批次间休息，避免限速
        if (page_num - start) % args.batch_size == 0 and page_num > start + 1:
            logger.info(f"    批次间休息 10s...")
            time.sleep(10)

        # 每页间短休息
        time.sleep(2)

    doc.close()

    if not all_entries:
        logger.error("未提取到任何产品!")
        sys.exit(1)

    logger.info(f"共提取 {len(all_entries)} 个原始产品")

    # 过滤非单体
    valid_entries = filter_valid_monomers(all_entries)
    logger.info(f"过滤后: {len(valid_entries)} 个单体产品")

    # 去重 (按 CAS 号)
    seen = {}
    deduped = []
    for e in valid_entries:
        cas = e.get("cas", "")
        if cas and cas != "null":
            if cas in seen:
                continue
            seen[cas] = e
        deduped.append(e)
    logger.info(f"去重后: {len(deduped)} 个单体产品")

    # 解析 SMILES
    resolved = []
    failed = []
    for i, e in enumerate(deduped):
        smi = resolve_smiles(e, cache)
        e["smiles"] = smi
        if smi:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                can_smi = Chem.MolToSmiles(mol, canonical=True)
                e["canonical_smiles"] = can_smi
                resolved.append(e)
                if (i + 1) % 20 == 0:
                    logger.info(f"  SMILES 解析: {i+1}/{len(deduped)} ({len(resolved)} 成功)")
            else:
                failed.append(e)
        else:
            failed.append(e)

    logger.info(f"SMILES 解析成功: {len(resolved)}/{len(deduped)}")

    # 保存结果
    df = pd.DataFrame(resolved)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    logger.info(f"商业单体已保存: {args.output} ({len(resolved)} 条)")

    # 保存 SMILES 缓存更新
    with open(args.cache, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    logger.info(f"SMILES 缓存已更新: {args.cache} ({len(cache)} 条)")

    # 失败列表
    if failed:
        fail_path = args.output.replace(".csv", "_failed.csv")
        pd.DataFrame(failed).to_csv(fail_path, index=False, encoding="utf-8-sig")
        logger.warning(f"SMILES 解析失败 {len(failed)} 条, 已保存: {fail_path}")


if __name__ == "__main__":
    main()
