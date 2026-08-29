# -*- coding: utf-8 -*-
"""
仂域 · 数字域 v1 · 探针实验
================================================================
不训练任何分类器。只用规则 + 锚点距离。规则不会因为洗牌而变。

三个口径（全部配正确对照 L0，否则无从谈"纠回来"）：

  口径1 探针 Probe
      prompt = "光在真空中的速度约为每秒"
      → 模型**自己**能否给出锚点值。测的是"知不知道"，不是"能不能纠"。
        作为基线：若探针都给不出，后面两个口径的低分就不能赖"纠不回来"。

  口径2 抗扰 Contamination
      prompt = "光在真空中的速度约为每秒3万千米。光在真空中的速度约为每秒"
      → 先注入错误，再问同一个问题，看续写是否**抵抗污染**回到锚点。
        这是"纠回来"的最弱形式：不需要显式纠错，只需不被带跑。

  口径3 回纠 Recovery
      prompt = "光在真空中的速度约为每秒3万千米。这句话里的数字是错的，正确数值是每秒"
      → 明确告知有错，看能否**主动找回**锚点。这是"纠回来"的强形式。

三态判定：
  RECOVERED    复现了锚点值，且未复现注入的错误值
  CONTAMINATED 复现了注入的错误值（被带跑）
  MISS         两者都未出现（无从判断）

复现性：固定 seed=42 + greedy 解码 + 全量跑 3 遍，三遍结果应逐位一致。
"""
import random

random.seed(42)          # ← 第一行就定种子（堵坑3）

import json
import os
import sys
import torch

torch.manual_seed(42)    # ← 同上

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import leyu_number_domain as D

# ----------------------------------------------------------------- 配置
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DEVICE = "cpu"
MAX_NEW = 24
N_REPEAT = 3                      # 全量跑 3 遍做复现性自检
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model():
    torch.manual_seed(42)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    mdl.to(DEVICE).eval()
    return tok, mdl


@torch.no_grad()
def gen(tok, mdl, prompt, max_new=MAX_NEW):
    """greedy 解码，保证可复现。"""
    ids = tok(prompt, return_tensors="pt").to(DEVICE)
    out = mdl.generate(**ids, max_new_tokens=max_new, do_sample=False,
                       pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)


def build_prompts(case):
    """
    三个口径的 prompt。L0（正确对照）在口径2/3 中把"错误陈述"换成"正确陈述"，
    这样三个口径对 L0 也有定义，对照组才真的成立。
    """
    stmt = case["statement"]          # 含 L0 原值 或 D1/D2/M 错误值
    probe = case["probe"]
    wrong_surf = None if case["error_type"] == "L0" else case["surface"]
    return {
        # 口径1：裸问
        "probe": probe,
        # 口径2：先注入错误，再原样重问一遍 —— 看是否被带跑
        "contam": f"{stmt}{probe}",
        # 口径3：明确告知有错，再重问 —— 看能否主动找回
        "recover": f"{stmt}这句话里的数字是错的。{probe}",
    }, wrong_surf


@torch.no_grad()
def _seq_logprob(mdl, ids):
    out = mdl(ids)
    lp = torch.log_softmax(out.logits[0, :-1].float(), dim=-1)
    return lp.gather(1, ids[0, 1:].unsqueeze(1)).squeeze(1).sum().item()


@torch.no_grad()
def cond_logprob(tok, mdl, context, continuation):
    """
    log P(continuation | context)。

    这是本实验最有信息量的一个量：**不受文本格式干扰**。
    base 模型裸续写会冒出 "____（判断对错）A. B. C." 这类试题模板，
    采样出来的文本三态判定会被格式噪声污染；而条件概率不会——
    它直接问："在这个上下文之后，模型给'正确数字'和'错误数字'各多少概率质量"。
    """
    ctx = tok(context, return_tensors="pt").input_ids.to(DEVICE)
    if not continuation:
        return 0.0
    cont = tok(continuation, return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)
    full = torch.cat([ctx, cont], dim=1)
    return _seq_logprob(mdl, full) - _seq_logprob(mdl, ctx)


def logodds(tok, mdl, context, correct_surf, wrong_surf):
    """
    对数优势比 Δ = logP(正确数字 | 上下文) − logP(错误数字 | 上下文)

      Δ > 0  模型更倾向锚点值 → 有纠回倾向
      Δ < 0  模型更倾向注入的错误值 → 被污染
      Δ ≈ 0  两者都不倾向（模型没主意）

    ★ 这是**连续量**，可以直接沿偏离梯度画曲线找临界点；
      而文本三态判定只有三档，画不出 R 的形状。
    """
    lp_c = cond_logprob(tok, mdl, context, correct_surf)
    lp_w = cond_logprob(tok, mdl, context, wrong_surf)
    return lp_c, lp_w, lp_c - lp_w


def judge(text, case, wrong_surf):
    """三态判定。"""
    ht = D.hit_test(text, BY_ID[case["anchor_id"]])
    if wrong_surf is None:
        # L0 对照组：没有注入错误，只判是否复现锚点
        return ("RECOVERED" if ht["hit_true"] else "MISS"), ht
    wv = D.surface_to_value(wrong_surf)
    hw = D.hit_wrong_test(text, wv) or (wrong_surf in text)
    if hw:
        state = "CONTAMINATED"
    elif ht["hit_true"]:
        state = "RECOVERED"
    else:
        state = "MISS"
    return state, ht


# 锚点查表（供 judge 使用）
BY_ID = {a["id"]: a for a in D.ANCHORS}
SURF = {a["id"]: a["surface"] for a in D.ANCHORS}
ALT = {a["id"]: a.get("alt", []) for a in D.ANCHORS}


def main():
    print("=" * 84)
    print("仂域 · 数字域 v1 · 探针实验")
    print("=" * 84)
    print(f"  模型      : {MODEL_NAME}（base，非 instruct）")
    print(f"  解码      : greedy（do_sample=False）")
    print(f"  种子      : random.seed(42) / torch.manual_seed(42)")
    print(f"  生成长度  : {MAX_NEW} token")
    print(f"  重复遍数  : {N_REPEAT}（用于复现性自检，三遍应逐位一致）")
    print(f"  锚点数    : {len(D.ANCHORS)}   错型: L0(对照) D1 D2 M   共 40 例")

    tok, mdl = load_model()
    print("模型就绪。\n")

    cases = D.build_cases()
    # 口径名 → 中文
    CAL = {"probe": "口径1·探针", "contam": "口径2·抗扰", "recover": "口径3·回纠"}

    all_runs = []
    for rep in range(N_REPEAT):
        torch.manual_seed(42)     # 每遍重置，若实现确定则三遍应完全相同
        random.seed(42)
        print(f"\n{'#' * 84}")
        print(f"# 第 {rep + 1} / {N_REPEAT} 遍")
        print(f"{'#' * 84}")
        run = {"rep": rep, "records": []}
        for ci, case in enumerate(cases):
            prompts, wrong_surf = build_prompts(case)
            rec = {
                "anchor_id": case["anchor_id"], "anchor_name": case["anchor_name"],
                "error_type": case["error_type"], "surface": case["surface"],
                "value_true": case["value_true"], "value_given": case["value_given"],
                "L": case["domain_measured"]["L"],
                "zone_measured": case["domain_measured"]["zone"],
                "zone_prior": case["domain_prior"]["zone"],
                "prompts": prompts, "out": {},
            }
            for ck in ("probe", "contam", "recover"):
                text = gen(tok, mdl, prompts[ck])
                state, ht = judge(text, case, wrong_surf)
                rec["out"][ck] = {"text": text, "state": state,
                                  "numbers": ht["numbers"], "hit_true": ht["hit_true"]}

            # ---- 口径4 · 概率对数优势比（连续量，不受生成格式干扰）----
            a = BY_ID[case["anchor_id"]]
            ctx = prompts["contam"]
            if wrong_surf is None:
                # L0 对照：没有注入错误，只记录 P(锚点 | 正确上下文) 作上界基线
                rec["logodds"] = {"lp_correct": cond_logprob(tok, mdl, ctx, a["surface"]),
                                  "lp_wrong": None, "delta": None}
            else:
                lp_c, lp_w, delta = logodds(tok, mdl, ctx, a["surface"], wrong_surf)
                rec["logodds"] = {"lp_correct": lp_c, "lp_wrong": lp_w, "delta": delta}
            run["records"].append(rec)
            if (ci + 1) % 10 == 0:
                print(f"  已完成 {ci + 1}/{len(cases)} 例")
        all_runs.append(run)

    # ------------------------------------------------- 复现性自检
    print("\n" + "=" * 84)
    print("复现性自检：三遍是否逐位一致")
    print("=" * 84)
    diff = 0
    for i in range(len(all_runs[0]["records"])):
        texts = []
        for r in all_runs:
            t = tuple(r["records"][i]["out"][k]["text"] for k in ("probe", "contam", "recover"))
            texts.append(t)
        if len(set(texts)) > 1:
            diff += 1
            print(f"  ✗ 第 {i} 例三遍不一致：{all_runs[0]['records'][i]['anchor_name']} "
                  f"{all_runs[0]['records'][i]['error_type']}")
    total = len(all_runs[0]["records"])
    print(f"  逐位一致: {total - diff}/{total}   "
          f"({'★ 完全一致，可复现' if diff == 0 else '存在不一致，需排查'})")

    # ------------------------------------------------- 汇总（用第 1 遍）
    recs = all_runs[0]["records"]
    print("\n" + "=" * 84)
    print("结果汇总 · 按错型")
    print("=" * 84)
    print(f"{'错型':<8}{'L(中位)':>9}{'口径':<12}{'RECOVERED':>11}{'CONTAM':>9}{'MISS':>7}"
          f"{'回纠率':>9}")
    print("-" * 84)
    summary = {}
    for et in ("L0", "D1", "D2", "M"):
        sub = [r for r in recs if r["error_type"] == et]
        Ls = sorted(r["L"] for r in sub if r["L"] is not None)
        Lmed = Ls[len(Ls) // 2] if Ls else None
        row = {}
        for ck in ("probe", "contam", "recover"):
            cnt = {"RECOVERED": 0, "CONTAMINATED": 0, "MISS": 0}
            for r in sub:
                cnt[r["out"][ck]["state"]] += 1
            n = len(sub)
            rate = cnt["RECOVERED"] / n if n else 0
            row[ck] = dict(cnt, n=n, rate=rate)
            print(f"{et:<8}{Lmed:>9.3f}{CAL[ck]:<12}{cnt['RECOVERED']:>11}"
                  f"{cnt['CONTAMINATED']:>9}{cnt['MISS']:>7}{rate:>8.1%}")
        summary[et] = {"L_median": Lmed, "n": len(sub), "by_caliber": row}
        print("-" * 84)

    # ------------------------------------------------- 口径4 汇总
    print("\n" + "=" * 84)
    print("口径4 · 概率对数优势比 Δ = logP(锚点|错误上下文) − logP(错误值|错误上下文)")
    print("=" * 84)
    print(f"{'错型':<8}{'n':>4}{'Δ均值':>10}{'Δ中位':>10}{'Δ>0 计数':>11}"
          f"{'Δ>0 占比':>10}   判定")
    print("-" * 84)
    LO = {}
    for et in ("D1", "D2", "M"):
        ds = [r["logodds"]["delta"] for r in recs
              if r["error_type"] == et and r["logodds"]["delta"] is not None]
        if not ds:
            continue
        mean = sum(ds) / len(ds)
        med = sorted(ds)[len(ds) // 2]
        pos = sum(1 for d in ds if d > 0)
        verdict = ("倾向锚点（可纠方向）" if pos / len(ds) >= 0.5
                   else "倾向错误值（被污染）")
        LO[et] = {"n": len(ds), "mean": mean, "median": med,
                  "pos": pos, "pos_rate": pos / len(ds)}
        summary[et]["logodds"] = LO[et]
        print(f"{et:<8}{len(ds):>4}{mean:>10.3f}{med:>10.3f}{pos:>11}"
              f"{pos / len(ds):>9.1%}   {verdict}")
    print("-" * 84)
    print("  注：Δ 是**连续量**，可沿偏离梯度直接找临界点；文本三态只有三档，画不出 R 的形状。")

    # ------------------------------------------------- 逐锚点明细
    print("\n" + "=" * 84)
    print("逐锚点明细（口径2·抗扰 —— 最接近'纠回来'的定义）")
    print("=" * 84)
    print(f"{'锚点':<14}{'真值':>11}  {'L0':<14}{'D1':<14}{'D2':<14}{'M':<14}")
    print("-" * 84)
    for a in D.ANCHORS:
        cells = []
        for et in ("L0", "D1", "D2", "M"):
            r = next((x for x in recs
                      if x["anchor_id"] == a["id"] and x["error_type"] == et), None)
            if r is None:
                cells.append("—")
                continue
            st = r["out"]["contam"]["state"]
            mark = {"RECOVERED": "✓纠回", "CONTAMINATED": "✗污染", "MISS": "·未现"}[st]
            cells.append(f"{r['surface']}:{mark}")
        print(f"{a['name']:<14}{a['value']:>11,.2f}  "
              + "".join(f"{c:<14}" for c in cells))
    print("-" * 84)

    # ------------------------------------------------- 导出 R
    print("\n" + "=" * 84)
    print("R 的导出")
    print("=" * 84)
    print("  R_域  = 最大还能纠回的偏离（以 L = log10(倍数) 为单位）")
    print()
    for et in ("D1", "D2", "M"):
        L = summary[et]["L_median"]
        r2 = summary[et]["by_caliber"]["contam"]["rate"]
        r3 = summary[et]["by_caliber"]["recover"]["rate"]
        print(f"    {et:<4} L≈{L:<7.3f}  抗扰回纠率 {r2:6.1%}   指令回纠率 {r3:6.1%}")

    # 找临界：最后一个回纠率 ≥50% 的错型，与第一个 <50% 的错型之间
    order = [("D1", summary["D1"]["L_median"]), ("D2", summary["D2"]["L_median"]),
             ("M", summary["M"]["L_median"])]
    last_ok = None
    first_bad = None
    for et, L in order:
        r = summary[et]["by_caliber"]["contam"]["rate"]
        if r >= 0.5:
            last_ok = (et, L, r)
        elif first_bad is None:
            first_bad = (et, L, r)
    print()
    if last_ok and first_bad:
        print(f"  边界落在 {last_ok[0]}(L≈{last_ok[1]:.3f}, 回纠率 {last_ok[2]:.1%}) "
              f"与 {first_bad[0]}(L≈{first_bad[1]:.3f}, 回纠率 {first_bad[2]:.1%}) 之间")
        print(f"  ⇒ R_抗扰 ∈ ({last_ok[1]:.3f}, {first_bad[1]:.3f})  "
              f"即偏离 {10**last_ok[1]:.2f}–{10**first_bad[1]:.2f} 倍之间")
    elif last_ok:
        print(f"  三个错型全部可纠（≥50%），R 至少覆盖到 L≈{last_ok[1]:.3f}"
              f"（{10**last_ok[1]:.1f} 倍）")
    else:
        print("  ★ 三个错型**无一**达到 50% 回纠率 —— 该模型在数字域上 R ≈ 0，")
        print("     即：不存在'自身能纠回的偏差'。这不否定域，只说明**纠的能力不在模型里**。")

    # 用口径4（连续量）给 R 定边界
    print()
    print("  ---- 以口径4（Δ 对数优势比）定边界 ----")
    seq = []
    for et in ("D1", "D2", "M"):
        if et in LO:
            seq.append((et, summary[et]["L_median"], LO[et]["mean"], LO[et]["pos_rate"]))
    for et, L, m, pr_ in seq:
        print(f"    {et:<4} L≈{L:<7.3f} Δ均值={m:>8.3f}  Δ>0占比={pr_:>6.1%}")
    # 临界：Δ 均值由正转负处
    last_pos = None
    first_neg = None
    for et, L, m, pr_ in seq:
        if m > 0:
            last_pos = (et, L, m)
        elif first_neg is None:
            first_neg = (et, L, m)
    print()
    if last_pos and first_neg:
        print(f"  ⇒ Δ 在 {last_pos[0]}(L≈{last_pos[1]:.3f}) 与 "
              f"{first_neg[0]}(L≈{first_neg[1]:.3f}) 之间穿越零点")
        print(f"  ⇒ R_概率 ∈ ({last_pos[1]:.3f}, {first_neg[1]:.3f})，"
              f"即 {10**last_pos[1]:.2f}–{10**first_neg[1]:.2f} 倍")
    elif last_pos:
        print(f"  ⇒ Δ 三档全为正，R 至少覆盖到 L≈{last_pos[1]:.3f}"
              f"（{10**last_pos[1]:.1f} 倍）")
    else:
        print("  ⇒ Δ 三档全为负：**模型在所有偏离档位上都更倾向错误值**。")
        print("     R_概率 = 0。含义不是'域不成立'，而是——")
        print("     **该模型自身不具备任何纠回能力，纠错必须完全由域承担。**")

    # ------------------------------------------------- 落盘
    out = {
        "meta": {"model": MODEL_NAME, "decode": "greedy", "seed": 42,
                 "max_new": MAX_NEW, "n_repeat": N_REPEAT,
                 "n_anchors": len(D.ANCHORS), "n_cases": len(cases)},
        "reproducibility": {"identical": total - diff, "total": total, "diff": diff},
        "summary": summary,
        "records": recs,
    }
    p_json = os.path.join(OUT_DIR, "result_v1.json")
    with open(p_json, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[saved] {p_json}")
    print("Done.")


if __name__ == "__main__":
    main()
