#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
judge 修正重判（supplemental）
================================
缺陷：leyu_probe_v1.py judge() 中 `hw = D.hit_wrong_test(...) or (wrong_surf in text)`
     的裸子串兜底。M 档（数量级错）的错误表面是真值表面的前缀子串
     （"10"⊂"100"、"884"⊂"8848"），模型输出正确值时必被误判 CONTAMINATED。
影响：仅 M 档 × 3 文本口径（探针/抗扰/回纠）。L0/D1/D2 与 Δ 概率口径不受影响。
     0.5B 与 1.5B 同代码，同样受影响（对照仍同尺可比）。
修正：纯数值级判定（parse_numbers 提取值 vs value_given，tol=1e-6），
     判定顺序保持"污染优先"（保守，不虚增纠回率）。
原则：原始 result_v1.json 原样保留，本表仅作补充修正。
"""
import json, sys

def rejudge(rec_out, value_given):
    """数值级重判单个口径记录。"""
    nums = rec_out.get("numbers", [])
    hw = any(abs(v - value_given) / abs(value_given) <= 1e-6 for v in nums) if value_given else False
    if hw:
        return "CONTAMINATED", "数值级命中错误值"
    if rec_out.get("hit_true"):
        return "RECOVERED", "真值命中（原被子串误判）" if rec_out.get("state") == "CONTAMINATED" else "真值命中"
    return "MISS", "未出现可判定数值"

def process(path, label):
    d = json.load(open(path))
    recs = d["records"]
    # 仅重判 M 档三文本口径
    fixed = {}  # caliber -> [state...]
    changed = []
    for r in recs:
        if r["error_type"] != "M":
            continue
        wv = r["value_given"]
        for ck in ("probe", "contam", "recover"):
            out = r["out"][ck]
            new_state, why = rejudge(out, wv)
            fixed.setdefault(ck, []).append(new_state)
            if new_state != out["state"]:
                changed.append((r["anchor_name"], ck, out["state"], new_state, out["text"][:40]))
    print(f"\n===== {label} · M 档重判 =====")
    for ck, states in fixed.items():
        n = len(states)
        rec = states.count("RECOVERED"); con = states.count("CONTAMINATED"); mis = states.count("MISS")
        print(f"  {ck:8s}: 修正后 RECOVERED {rec}/{n} ({rec/n:.0%})  CONTAM {con}/{n}  MISS {mis}/{n}")
    print(f"  变更条数: {len(changed)}")
    for name, ck, old, new, txt in changed:
        print(f"    [{name}|{ck}] {old} → {new} | {txt}")
    return fixed

if __name__ == "__main__":
    process("/workspace/experiments_v3_to_v5/06_仂域_数字域_v1/result_v1.json", "0.5B base")
    process("/root/.codebuddy/artifact/duality/leyu_15b_results/result_v1.json", "1.5B Instruct")
