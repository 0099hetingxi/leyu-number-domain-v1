#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""门控架构 v1：max_new=24（首轮，含两处缺陷） vs max_new=96（修正后）全量对照"""
import json, os, collections

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["L0", "D1", "D2", "M"]


def load(path):
    return json.load(open(os.path.join(HERE, path)))


def group(d):
    """按 (anchor_id, error_type) 聚合各遍"""
    cases = collections.OrderedDict()
    for r in d["runs"]:
        for rec in r["records"]:
            key = (rec["anchor_id"], rec["error_type"])
            cases.setdefault(key, {"meta": rec, "reps": []})["reps"].append(rec)
    return cases


def stat(cases, fn):
    out = collections.defaultdict(lambda: [0, 0])
    for (aid, et), c in cases.items():
        out[et][1] += 1
        if fn(c["reps"]):
            out[et][0] += 1
    return {k: (out[k][0], out[k][1]) for k in ORDER}


def allof(rs, key):
    return all(r[key] == "RECOVERED" for r in rs)


def analyze(d, label):
    cases = group(d)
    res = {"label": label, "n": len(cases), "rep": d["meta"]["n_repeat"]}
    res["base"] = stat(cases, lambda rs: all(r["baseline"]["state"] == "RECOVERED" for r in rs))
    res["gstrict"] = stat(cases, lambda rs: all(r["gated"]["state"] == "RECOVERED" for r in rs))
    res["geff"] = stat(cases, lambda rs: all(r["gated"]["state_effective"] == "RECOVERED" for r in rs))

    # 门控动作
    act = collections.Counter()
    act_ok = collections.Counter()
    act_by_tier = collections.defaultdict(collections.Counter)
    for (aid, et), c in cases.items():
        for r in c["reps"]:
            a = r["gated"]["action"]
            act[a] += 1
            act_by_tier[et][a] += 1
            if r["gated"]["state_effective"] == "RECOVERED":
                act_ok[a] += 1
    res["act"], res["act_ok"], res["act_by_tier"] = act, act_ok, act_by_tier

    # 转化
    up = down = good = bad = 0
    for (aid, et), c in cases.items():
        b = all(r["baseline"]["state"] == "RECOVERED" for r in c["reps"])
        g = all(r["gated"]["state_effective"] == "RECOVERED" for r in c["reps"])
        if not b and g: up += 1
        elif b and not g: down += 1
        elif b and g: good += 1
        else: bad += 1
    res["conv"] = {"up": up, "down": down, "good": good, "bad": bad}

    # 漏检：门控放行（行）但终态仍污染
    miss = []
    for (aid, et), c in cases.items():
        for i, r in enumerate(c["reps"]):
            if r["gated"]["action"] == "行" and r["gated"]["state_effective"] != "RECOVERED":
                miss.append((et, r["anchor_name"], r["value_given"], r["value_true"]))
    res["miss"] = miss

    # 复现性
    cons = sum(1 for c in cases.values()
               if len({r["baseline"]["state"] for r in c["reps"]}) == 1
               and len({r["gated"]["state_effective"] for r in c["reps"]}) == 1
               and len({r["gated"]["action"] for r in c["reps"]}) == 1)
    res["cons"] = (cons, len(cases))

    # 失败清单
    fails = []
    for (aid, et), c in cases.items():
        if not all(r["gated"]["state_effective"] == "RECOVERED" for r in c["reps"]):
            r0 = c["reps"][0]
            fails.append((et, r0["anchor_name"], r0["value_given"], r0["value_true"], r0["L"],
                          [r["baseline"]["state"] for r in c["reps"]],
                          [r["gated"]["state_effective"] for r in c["reps"]],
                          [r["gated"]["action"] for r in c["reps"]]))
    res["fails"] = sorted(fails)
    return res


A = analyze(load("gated_v1_result_maxnew24.json"), "首轮 max_new=24（含缺陷）")
B = analyze(load("gated_v1_result_maxnew96.json"), "修正后 max_new=96")

print("=" * 84)
print("仂域门控架构 v1 · 两轮全量对照（40 例 × 3 遍，3 遍全对才计入）")
print("=" * 84)
print(f"{'档':<5}{'baseline':^24}{'gated严格':^24}{'gated架构':^24}")
print(f"{'':<5}{'24':^12}{'96':^12}{'24':^12}{'96':^12}{'24':^12}{'96':^12}")
print("-" * 84)
for k in ORDER:
    row = f"{k:<5}"
    for key in ["base", "gstrict", "geff"]:
        for R in (A, B):
            v, n = R[key][k]
            row += f"{v}/{n}({100*v//n:>2}%)".rjust(12)
    print(row)

print()
print("=" * 84)
print("门控动作 → 架构口径终态（拦截成功率）")
print("=" * 84)
print(f"{'动作':<8}{'首轮次数':>10}{'首轮成功':>10}{'':>4}{'修正次数':>10}{'修正成功':>10}{'':>4}{'成功率 24→96':>16}")
for a in ["行", "待", "撤"]:
    t1, o1 = A["act"].get(a, 0), A["act_ok"].get(a, 0)
    t2, o2 = B["act"].get(a, 0), B["act_ok"].get(a, 0)
    r1 = f"{100*o1//t1}%" if t1 else "—"
    r2 = f"{100*o2//t2}%" if t2 else "—"
    print(f"{a:<8}{t1:>10}{o1:>10}{'':>4}{t2:>10}{o2:>10}{'':>4}{r1+' → '+r2:>16}")

print()
print("=" * 84)
print("逐例转化与复现性")
print("=" * 84)
for R in (A, B):
    c = R["conv"]
    print(f"{R['label']:<28} 修好 {c['up']:>2} | 修坏 {c['down']:>2} | 本就对 {c['good']:>2} | 仍错 {c['bad']:>2} "
          f"| 复现性 {R['cons'][0]}/{R['cons'][1]}")

print()
print("=" * 84)
print("漏检（门控判『行』放行，但终态实际仍污染）")
print("=" * 84)
for R in (A, B):
    print(f"{R['label']}: {len(R['miss'])} 条例×遍")
    cnt = collections.Counter((m[0], m[1]) for m in R["miss"])
    for (et, name), n in sorted(cnt.items()):
        print(f"    [{et}] {name}  ×{n}")

print()
print("=" * 84)
print("失败清单（架构口径下未修好）——修正后")
print("=" * 84)
for f in B["fails"]:
    et, name, vg, vt, L, bl, ge, acts = f
    print(f"  [{et}] {name:<10} 声明值 {vg} vs 真值 {vt}  (L={L:.4f})")
    print(f"       baseline={set(bl)}  动作={acts}  终态={set(ge)}")
