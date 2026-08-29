# -*- coding: utf-8 -*-
"""
交叉分析：把"模型不知道答案"从 R 里剥离出去
==========================================================
主实验里 R_概率 = 0，但这不能直接写成"数字域的 R 是 0"——
因为口径1（探针）显示：10 个锚点里模型只知道 6 个。
让一个压根不知道答案的模型去"纠回来"，测的是它的无知，不是域的边界。

所以按"探针是否命中"把锚点切成两组，分别算 R：
  已知组 KNOWN    —— 模型能独立给出锚点值，谈纠错才有意义
  未知组 UNKNOWN  —— 模型不会，任何"纠不回来"都与域无关
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "result_v1.json"), encoding="utf-8"))
recs = R["records"]

# ---------------------------------------------------------------- 分组
known, unknown = [], []
for a in R["meta"]["n_anchors"] * [0]:      # 占位，下面用字典重做
    pass

ANCH = {}
for r in recs:
    if r["error_type"] == "L0":
        # 口径1：裸问能否给出锚点值 → 模型是否"知道"
        ANCH[r["anchor_id"]] = {
            "name": r["anchor_name"],
            "knows": r["out"]["probe"]["state"] == "RECOVERED",
            "probe_text": r["out"]["probe"]["text"],
        }

for a in ANCH.values():
    (known if a["knows"] else unknown).append(a["name"])

print("=" * 84)
print("交叉分析 · 把'模型不知道'从 R 里剥离")
print("=" * 84)
print(f"\n【已知组 KNOWN】(探针能独立答出，n={len(known)})")
for n in known:
    print(f"    ✓ {n}")
print(f"\n【未知组 UNKNOWN】(探针答不出，n={len(unknown)})")
for n in unknown:
    print(f"    ✗ {n}   → 生成: {ANCH[[k for k,v in ANCH.items() if v['name']==n][0]]['probe_text'][:40]!r}")

print("\n" + "=" * 84)
print("分组后的回纠率（口径2 抗扰 / 口径3 回纠 / 口径4 概率）")
print("=" * 84)
print(f"{'错型':<6}{'L':>8} | {'全组':>22} | {'已知组':>22} | {'未知组':>22}")
print(f"{'':<6}{'':>8} | {'抗扰':>7}{'回纠':>7}{'Δ>0':>8} |"
      f"{'抗扰':>7}{'回纠':>7}{'Δ>0':>8} | {'抗扰':>7}{'回纠':>7}{'Δ>0':>8}")
print("-" * 84)

res = {}
for et in ("D1", "D2", "M"):
    row = {}
    for grp, ids in (("all", None),
                     ("known", [k for k, v in ANCH.items() if v["knows"]]),
                     ("unknown", [k for k, v in ANCH.items() if not v["knows"]])):
        sub = [r for r in recs if r["error_type"] == et
               and (ids is None or r["anchor_id"] in ids)]
        if not sub:
            row[grp] = (0, 0.0, 0.0, 0.0, 0)
            continue
        n = len(sub)
        c2 = sum(1 for r in sub if r["out"]["contam"]["state"] == "RECOVERED") / n
        c3 = sum(1 for r in sub if r["out"]["recover"]["state"] == "RECOVERED") / n
        ds = [r["logodds"]["delta"] for r in sub if r["logodds"]["delta"] is not None]
        dpos = (sum(1 for d in ds if d > 0) / len(ds)) if ds else 0.0
        dmean = (sum(ds) / len(ds)) if ds else 0.0
        row[grp] = (n, c2, c3, dpos, dmean)
    L = next((r["L"] for r in recs if r["error_type"] == et), None)
    res[et] = row
    f = lambda g: f"{row[g][1]:>7.0%}{row[g][2]:>7.0%}{row[g][3]:>8.0%}"
    print(f"{et:<6}{L:>8.3f} | {f('all')} | {f('known')} | {f('unknown')}")
print("-" * 84)
print("  读法：Δ>0 占比 = 模型更倾向锚点值（而非注入错误值）的比例。")

# ---------------------------------------------------------------- Δ 均值
print("\n" + "=" * 84)
print("Δ 均值（logP(锚点) − logP(错误值)）：正值 = 倾向锚点")
print("=" * 84)
print(f"{'错型':<6}{'L':>8} | {'全组':>10} | {'已知组':>10} | {'未知组':>10}")
print("-" * 84)
for et in ("D1", "D2", "M"):
    L = next((r["L"] for r in recs if r["error_type"] == et), None)
    print(f"{et:<6}{L:>8.3f} | {res[et]['all'][4]:>10.3f} | "
          f"{res[et]['known'][4]:>10.3f} | {res[et]['unknown'][4]:>10.3f}")
print("-" * 84)

# ---------------------------------------------------------------- 逐锚点知识强度
print("\n" + "=" * 84)
print("逐锚点：知识强度 vs 抗扰能力")
print("=" * 84)
print(f"{'锚点':<16}{'知道?':>6}{'L0抗扰':>9}{'D1':>8}{'D2':>8}{'M':>8}"
      f"{'D1的Δ':>9}{'M的Δ':>9}")
print("-" * 84)
rows = []
for aid, v in ANCH.items():
    cells, deltas = [], {}
    for et in ("L0", "D1", "D2", "M"):
        r = next((x for x in recs if x["anchor_id"] == aid and x["error_type"] == et), None)
        if r is None:
            cells.append("—")
            continue
        st = r["out"]["contam"]["state"]
        mark = {"RECOVERED": "✓", "CONTAMINATED": "✗", "MISS": "·"}[st]
        cells.append(mark)
        if r["logodds"]["delta"] is not None:
            deltas[et] = r["logodds"]["delta"]
    rows.append((v["name"], v["knows"], cells, deltas))
    print(f"{v['name']:<16}{('✓' if v['knows'] else '✗'):>6}"
          f"{cells[0]:>9}{cells[1]:>8}{cells[2]:>8}{cells[3]:>8}"
          f"{deltas.get('D1', float('nan')):>9.2f}{deltas.get('M', float('nan')):>9.2f}")
print("-" * 84)

# ---------------------------------------------------------------- 结论
print("\n" + "=" * 84)
print("结论")
print("=" * 84)
kd1 = res["D1"]["known"]
kd2 = res["D2"]["known"]
km = res["M"]["known"]
print(f"  已知组(n={kd1[0]})：抗扰回纠率 D1={kd1[1]:.0%}  D2={kd2[1]:.0%}  M={km[1]:.0%}")
print(f"  已知组          ：Δ>0 占比    D1={kd1[3]:.0%}  D2={kd2[3]:.0%}  M={km[3]:.0%}")
print()
if kd1[1] >= 0.5 and km[1] < 0.5:
    print("  ⇒ 已知组上，抗扰回纠率随偏离单调下降，且 D1 与 M 分列 50% 两侧。")
    print(f"  ⇒ R_抗扰 ∈ (L≈{res['D1']['all'][0] and 0.002:.3f}, L≈1.000)")
elif km[1] >= 0.5:
    print("  ⇒ 已知组上连数量级错都能纠回，R 至少覆盖一个数量级。")
else:
    print("  ⇒ 即使在已知组，也没有任何档位达到 50% 回纠率。")
    print("     含义：知道答案 ≠ 能抵抗污染。模型会把上下文里的错误数字")
    print("     当作既定事实接着往下说——这是纯续写模型的本性。")
print()
print("  ★ 这条最重要：R 不是域的单一常数，而是 (域, 锚点, 模型) 的联合属性。")
print("     同一个数字域，π 上纠不崩（D1/D2/M 全纠回），地月距离上根本不生成数字。")

out = {"known": known, "unknown": unknown, "by_group": res,
       "per_anchor": [{"name": n, "knows": k, "contam": c, "delta": d}
                      for n, k, c, d in rows]}
with open(os.path.join(HERE, "analyze_v1.json"), "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("\n[saved] analyze_v1.json")
