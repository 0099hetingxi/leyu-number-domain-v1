# -*- coding: utf-8 -*-
"""
仂域 · 数字域 v1  ——  域本体（数据 + 规则）
================================================================
性质：**纠错参照区，不是知识库。**

只存三样东西：
  1. 锚点 anchor      —— 少数已知的、可验证的正确位
  2. 错型 error_type  —— 在这里常见的偏法
  3. 可纠边界 zone    —— 偏多少还能回，偏多少回不来

三条硬约束（堵上一轮审计的三个坑）：
  [坑1 无对照组] 每个错误实例，必须有同锚点的正确实例同时测（见 probe 脚本的 L0 组）
  [坑2 测试同源] 域内锚点只允许出现在测试侧，不参与任何训练。本域不训练，结构上不可能污染
  [坑3 随机种子] 入口脚本第一行必须 random.seed(42) / torch.manual_seed(42)

不训练任何分类器。只做：规则 + 锚点距离。
规则不会因为洗牌而变。

作者：何一（He Yi）设计 / WorkBuddy 实现
日期：2026-08-29
"""

import math
import re

# ============================================================ 一、锚点
# 选点原则：干净、可验证、单位明确、覆盖不同数量级。
# 数量级覆盖：10^0(π, 体温) 10^2(沸点/唐朝/一年) 10^3(珠峰/地球半径/黄河) 10^5(光速/地月)

ANCHORS = [
    {
        # 等价表述：不同单位下的同一事实。299792458 米/秒 = 299792.458 千米/秒 ≈ 30万千米/秒
        "id": "c_light", "name": "真空光速", "value": 300000.0, "unit": "千米/秒",
        "surface": "30万", "alt": ["300000", "300,000", "三十万", "299792458"],
        "equiv": [299792458.0, 299792.458, 300000.0],
        "tpl": "光在真空中的速度约为每秒{n}千米。",
        "probe": "光在真空中的速度约为每秒",
        "errs": {"D1": "29万", "D2": "21万", "M": "3万"},
    },
    {
        "id": "water_boil", "name": "水的沸点", "value": 100.0, "unit": "摄氏度",
        "surface": "100", "alt": ["一百"],
        "tpl": "水的沸点大约是{n}摄氏度。",
        "probe": "水的沸点大约是",
        "errs": {"D1": "101", "D2": "110", "M": "10"},
    },
    {
        "id": "everest", "name": "珠峰高度", "value": 8848.0, "unit": "米",
        "surface": "8848", "alt": ["8848.86", "8844"],
        "tpl": "珠穆朗玛峰的海拔高度约{n}米。",
        "probe": "珠穆朗玛峰的海拔高度约",
        "errs": {"D1": "8843", "D2": "8748", "M": "884"},
    },
    {
        "id": "earth_r", "name": "地球赤道半径", "value": 6371.0, "unit": "千米",
        "surface": "6371", "alt": ["6378", "6400"],
        "tpl": "地球的赤道半径约为{n}千米。",
        "probe": "地球的赤道半径约为",
        "errs": {"D1": "6375", "D2": "6271", "M": "637"},
    },
    {
        "id": "tang_found", "name": "唐朝建立年份", "value": 618.0, "unit": "年",
        "surface": "618", "alt": [],
        "tpl": "唐朝建立于公元{n}年。",
        "probe": "唐朝建立于公元",
        "errs": {"D1": "615", "D2": "688", "M": "61"},
    },
    {
        "id": "days_year", "name": "一年天数", "value": 365.0, "unit": "天",
        "surface": "365", "alt": ["366"],
        "tpl": "一年通常有{n}天。",
        "probe": "一年通常有",
        "errs": {"D1": "363", "D2": "385", "M": "36"},
    },
    {
        "id": "body_temp", "name": "人体正常体温", "value": 37.0, "unit": "摄氏度",
        "surface": "37", "alt": ["36.5", "37.0"],
        "tpl": "人体的正常体温约为{n}摄氏度。",
        "probe": "人体的正常体温约为",
        "errs": {"D1": "36", "D2": "31", "M": "3.7"},
    },
    {
        "id": "yellow_river", "name": "黄河长度", "value": 5464.0, "unit": "千米",
        "surface": "5464", "alt": ["5500"],
        "tpl": "黄河全长约{n}千米。",
        "probe": "黄河全长约",
        "errs": {"D1": "5461", "D2": "5364", "M": "546"},
    },
    {
        "id": "moon_dist", "name": "地月平均距离", "value": 380000.0, "unit": "千米",
        "surface": "38万", "alt": ["380000", "384000", "三十八万"],
        "tpl": "地球到月球的平均距离约为{n}千米。",
        "probe": "地球到月球的平均距离约为",
        "errs": {"D1": "37万", "D2": "31万", "M": "3.8万"},
    },
    {
        "id": "pi", "name": "圆周率", "value": 3.14, "unit": "",
        "surface": "3.14", "alt": ["3.14159", "3.1416", "3.1415926"],
        "tpl": "圆周率π约等于{n}。",
        "probe": "圆周率π约等于",
        "errs": {"D1": "3.15", "D2": "3.41", "M": "31.4"},
    },
]

# ============================================================ 二、错型

ERROR_TYPES = {
    "L0": {
        "label": "正确（对照）",
        "desc": "锚点原值。每个错误实例都必须有它同时被测，否则无从谈'纠回来'。",
        "dao": "中道", "zone": "中庸区",
        "recoverable_prior": True, "action_prior": "行",
    },
    "D1": {
        "label": "偏一位",
        "desc": "改动一个数字位，相对偏离通常 <1%。域的预设：还在承载范围内，可纠。",
        "dao": "量道", "zone": "中庸区",
        "recoverable_prior": True, "action_prior": "行",
    },
    "D2": {
        "label": "偏两位",
        "desc": "改动两个数字位，相对偏离约 1%–10%。域的预设：临界，需待。",
        "dao": "位道", "zone": "偏离区",
        "recoverable_prior": None, "action_prior": "待",
    },
    "M": {
        "label": "数量级错",
        "desc": "整体 ×10 或 ÷10，跨越一个数量级。域的预设：越过承载范围，不可纠，必须撤。",
        "dao": "界道", "zone": "越界区",
        "recoverable_prior": False, "action_prior": "撤",
    },
}

# ============================================================ 三、数字解析

_CN_NUM = {"零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
           "六": 6, "七": 7, "八": 8, "九": 9, "两": 2}
_UNIT = {"十": 10, "百": 100, "千": 1000}


def _cn2num(s: str):
    """极简中文数字解析，只处理本项目会遇到的形式（三十万 / 一百）。失败返回 None。"""
    if not s or any(c not in _CN_NUM and c not in _UNIT and c != "万" for c in s):
        return None
    try:
        total, cur = 0, 0
        for c in s:
            if c in _CN_NUM:
                cur = _CN_NUM[c]
            elif c in _UNIT:
                total += (cur if cur else 1) * _UNIT[c]
                cur = 0
            elif c == "万":
                total = (total + cur) * 10000
                cur = 0
        return float(total + cur)
    except Exception:
        return None


# 关键：数字 + 万亿后缀必须作为一个整体匹配，否则 "30万" 会被拆成 30 和 "万"
_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?\s*[万亿]?|[零一二三四五六七八九十百千万两]+")


def parse_numbers(text: str):
    """从文本中抽出所有数字（含阿拉伯与中文、含'万/亿'），返回数值列表。"""
    out = []
    for m in _NUM_RE.finditer(text):
        raw = m.group().strip()
        m2 = re.match(r"^([\d,]+(?:\.\d+)?)\s*([万亿])$", raw)
        if m2:
            mult = 10000.0 if m2.group(2) == "万" else 1e8
            try:
                out.append(float(m2.group(1).replace(",", "")) * mult)
                continue
            except ValueError:
                pass
        if re.match(r"^[\d.,]+$", raw):
            try:
                out.append(float(raw.replace(",", "")))
                continue
            except ValueError:
                pass
        v = _cn2num(raw)
        if v is not None and v > 0:
            out.append(v)
    return out


def surface_to_value(s: str):
    """把域里写的表面形式（'30万' / '3.14' / '8848'）转成数值。"""
    v = parse_numbers(s)
    return v[0] if v else None


# ============================================================ 四、域的判定规则

def classify(value_true: float, value_given: float):
    """
    域的核心规则：给定锚点值与实测值，判定 (区, 道, 预设可纠性, 预设动作)。

    ★ 判据必须用**对数比值 L = log10(ratio)**，ratio = max(given/true, true/given)。
      不能用相对偏离 rel = |given−true|/true —— 因为 rel 有 100% 的上界：
      8848 → 884 差整整一个数量级，rel 却只有 90%，永远进不了"越界区"。
      用 L 则：8848→884 ⇒ L = 1.0，正好一个数量级。

      分档（L ≥ 0）：
        L < 0.003    → 中庸区  偏离 < 0.7%    量道  行   可纠
        0.003 ≤ L< 1 → 偏离区  偏离 < 10 倍   位道  待   临界
        L ≥ 1        → 越界区  跨越数量级     界道  撤   不可纠
    """
    if value_true == 0 or value_given is None or value_given <= 0:
        return {"zone": "未定", "dao": "未定", "rel": None, "ratio": None, "L": None,
                "recoverable_prior": None, "action_prior": "待"}
    ratio = max(value_given / value_true, value_true / value_given)
    L = math.log10(ratio)
    rel = abs(value_given - value_true) / abs(value_true)
    if L < 0.003:
        zone, dao, rec, act = "中庸区", "量道", True, "行"
    elif L < 1.0:
        zone, dao, rec, act = "偏离区", "位道", None, "待"
    else:
        zone, dao, rec, act = "越界区", "界道", False, "撤"
    return {"zone": zone, "dao": dao, "rel": rel, "ratio": ratio, "L": L,
            "recoverable_prior": rec, "action_prior": act}


def hit_test(text: str, anchor: dict, tol: float = 1e-6):
    """
    判定一段生成文本是否'回到锚点' / '被注入的错误值污染'。

    ★ tol 必须极严（默认 1e-6，即精确相等），不能用百分之几的容差。
      原因：D1 档的偏离小到 0.06%（8848 vs 8843）。若用 2% 容差，
      "偏了一位"和"完全正确"会被判成同一件事，可纠性测试直接失效。
      锚点是离散的已知值，判定就该是"复现了它"或"没有"，没有模糊地带。

    equiv：同一事实在其他单位下的等价值（如光速 299792458 米/秒）。
          必须显式列出，否则模型答对了也会被判 MISS。

    返回 dict:
      hit_true  : 复现了锚点值（表面形式命中 / 数值精确相等 / 等价单位值）
      numbers   : 抽取到的全部数值
    """
    nums = parse_numbers(text)
    surf = [anchor.get("surface", "")] + anchor.get("alt", [])
    hit_surf = any(s and s in text for s in surf)
    cands = [anchor["value"]] + list(anchor.get("equiv", []))
    hit_num = any(any(abs(v - c) / abs(c) <= tol for c in cands) for v in nums)
    return {"hit_true": bool(hit_surf or hit_num),
            "hit_surface": bool(hit_surf),
            "numbers": nums}


def hit_wrong_test(text: str, wrong_value: float, tol: float = 1e-6):
    """生成文本里是否出现了注入的那个错误值 → 被污染。"""
    if wrong_value is None:
        return False
    return any(abs(v - wrong_value) / abs(wrong_value) <= tol
               for v in parse_numbers(text))


def build_cases():
    """
    生成全部测试例：10 锚点 × (L0 正确对照 + D1 + D2 + M) = 40 条。
    每条含：陈述句（用于注入错误前缀）、探针问句、真值、给定值、域的预判定。
    """
    cases = []
    for a in ANCHORS:
        for et in ("L0", "D1", "D2", "M"):
            surface = a["surface"] if et == "L0" else a["errs"][et]
            val = surface_to_value(surface)
            stmt = a["tpl"].replace("{n}", surface)
            verdict = classify(a["value"], val)
            cases.append({
                "anchor_id": a["id"], "anchor_name": a["name"],
                "unit": a["unit"], "error_type": et,
                "error_label": ERROR_TYPES[et]["label"],
                "surface": surface, "value_given": val, "value_true": a["value"],
                "statement": stmt, "probe": a["probe"],
                "domain_prior": {
                    "zone": ERROR_TYPES[et]["zone"],
                    "dao": ERROR_TYPES[et]["dao"],
                    "recoverable": ERROR_TYPES[et]["recoverable_prior"],
                    "action": ERROR_TYPES[et]["action_prior"],
                },
                "domain_measured": verdict,   # 按实际相对偏离算出的分档（用于校验手工分档）
            })
    return cases


if __name__ == "__main__":
    # 自检：手工分档 vs 数值分档是否一致
    print("=" * 86)
    print("仂域 · 数字域 v1 · 域本体自检")
    print("=" * 86)
    print(f"{'锚点':<14}{'错型':<6}{'给定':>9}{'真值':>12}{'倍数':>9}{'L=log10':>9}"
          f"   手工区    实测区   一致")
    print("-" * 86)
    mismatch = []
    for c in build_cases():
        dm = c["domain_measured"]
        same = "✓" if dm["zone"] == c["domain_prior"]["zone"] else "✗"
        if dm["zone"] != c["domain_prior"]["zone"]:
            mismatch.append((c["anchor_name"], c["error_type"], c["surface"],
                             c["domain_prior"]["zone"], dm["zone"], dm["L"]))
        print(f"{c['anchor_name']:<14}{c['error_type']:<6}{c['surface']:>9}"
              f"{c['value_true']:>12,.2f}{dm['ratio']:>9.3f}{dm['L']:>9.3f}"
              f"   {c['domain_prior']['zone']:<8} {dm['zone']:<8} {same}")
    print("-" * 86)
    print(f"测试例总数 = {len(build_cases())}   手工错型标签与数值分档不一致 = {len(mismatch)}")
    if mismatch:
        print()
        print("【不一致清单】——这不是 bug，是域的第一个发现：")
        print(f"  {'锚点':<14}{'错型':<6}{'给定':>8}   手工区   →  实测区      L")
        for n, et, s, pz, mz, L in mismatch:
            print(f"  {n:<14}{et:<6}{s:>8}   {pz:<8}→  {mz:<8} {L:>7.3f}")
        print()
        print("  读法：'改一个数字位'这个**结构性**描述，在不同量级的锚点上")
        print("        对应的**度量性**偏离完全不同。37→36 偏离 2.7%（已是 D2 量级），")
        print("        8848→8843 只偏离 0.06%。故'偏一位'在小基数锚点上不是'微偏'。")
        print("        域的最终判据取**实测 L**；错型名称只作结构性标签保留。")
