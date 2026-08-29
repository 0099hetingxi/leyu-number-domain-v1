#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
"""
仂域求存门控架构 v1（Leyu Gated Pipeline v1）
================================================================
架构重排（2026-08-29 何一指令）：

  任务先进仂域求存，经过分词库、嵌入层、神经网络，
  进入端不用仂感头；输出那里加门控，仂感头放进仂域里。

  旧路（已实测走死，07-19 → 08-29）：
    仂感头插在模型进入端/内部 → 自参照 → 只能测相对异常 → R_model = 0
    证据：v4.0 34.6%、v5.0 九处缺陷（100% 检出被推翻）、
          1.5B-Instruct 抗扰 30/30/30（探针 10/10 全知道也纠不回）

  新路（本架构）：
    任务 → 仂域·求存预检 → 分词 → 嵌入 → 神经网络（干净主干，不插头）
         → 输出 → 仂域·门控（仂感头在域内，外置锚 + L 判据）
         → 三态：行（放行）/ 待（重问，二轮输出再过仂域门控）/ 撤（拦截，域供给正确值）
    ★ 门控决策本身也在仂域内（adjudicate）：管线无判定逻辑，只执行仂域裁决。

四个部件：
  [1] LeyuDomain.scan_task    任务求存预检：任务先在仂域过一遍，
                              扫出触及的锚点、任务自带数字的 L 分档
  [2] CleanBackbone           干净主干：分词 → 嵌入 → 神经网络 → logits
                              不加 hook、不插仂感头（进入端不用仂感头）
  [3] LeyuDomain.sense        仂感头（在仂域里）：对生成文本按外置锚做
                              数值级 L 判据（无子串兜底——judge 老坑不带入）
  [4] LeyuDomain.adjudicate   门控（在仂域里）：仂感头判定 + 三态裁决 + 域供给；
                              重问二轮再过仂域门控，全程决策不出域

与审计九坑的对应（不重蹈）：
  坑「NEUTRAL 恒 0 无分支」→ 门控三态显式分支，每例必有终态
  坑「自参照 R=0」        → 判定只用外置锚，不用模型内生量
  坑「随机基线混入」       → 域不训练、规则固定、判定不依赖随机性
  坑「子串误判」           → 仂感头纯数值级（tol=1e-6）

作者：何一（He Yi）架构 / WorkBuddy 实现
日期：2026-08-29
"""
import argparse, json, math, os, random, re, sys, time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import leyu_number_domain as D

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
# MAX_NEW=24 是 0.5B/1.5B 前序实验的历史遗留值。实测发现它对 baseline 系统性不利：
# 六成 L0 输出停在选择题选项上（"A. 98摄氏度 B. 102摄氏度 C. "），答案尚未生成即被截断；
# 而门控判"撤"时由域直接供值、不受截断影响。同尺对照下这会人为抬高门控的相对提升。
# 改为 96，让 baseline 有公平机会说完（见报告"截断偏差"一节）。
MAX_NEW = 96
N_REPEAT = 3
SEED = 42

CAL = {"probe": "探针", "contam": "抗扰", "recover": "回纠"}

# 锚点触发词：仂感头的语义归属闸门——文本谈到锚点（触发词同句）才激活判定，
# 无关数字（续写里的"一分钟"之类）不参与锚点的 L 比对。
TRIGGERS = {
    "c_light":     ["光速", "真空", "每秒"],
    "water_boil":  ["沸点", "沸腾"],
    "everest":     ["珠峰", "珠穆朗玛", "海拔"],
    "earth_r":     ["赤道半径", "地球半径", "赤道"],
    "tang_found":  ["唐朝", "唐代"],
    "days_year":   ["一年", "天数"],
    "body_temp":   ["体温"],
    "yellow_river": ["黄河"],
    "moon_dist":   ["地月", "月球"],
    "pi":          ["圆周率", "π"],
}

_SENT_SPLIT = re.compile(r"[。！？\n；;]")


# ============================================================ [1][3][4] 仂域
class LeyuDomain:
    """仂域：锚点库 + 错型 + 仂感头 + 求存预检 + R 账本。全部外置，不训练。"""

    def __init__(self):
        self.anchors = {a["id"]: a for a in D.ANCHORS}
        self.ledger = []          # R 账本：每次门控判定的记录
        self.r_by_anchor = {}     # R(域,锚点,模型) 的实测累积

    # 物理单位词表（长度降序）。"千米"的"千"会被 _NUM_RE 当成数值 1000，
    # 实测 "30万千米" → [300000.0, 1000.0]、"6371千米" → [6371.0, 1000.0]，
    # 导致 L0（无错）任务被判 L=2.477 越界、门控无谓动作。剥离单位词再解析。
    _UNIT_WORDS = [
        "千米/秒", "米/秒", "公里/小时", "千瓦时", "平方公里", "平方厘米", "立方千米",
        "立方米", "厘米", "毫米", "毫升", "毫克", "千克", "公斤", "千瓦", "摄氏度",
        "光年", "天文单位", "海里", "英里", "英尺", "英寸", "赫兹", "帕斯卡",
        "焦耳", "安培", "伏特", "欧姆", "小时", "分钟", "千米", "公里", "分米",
        "微米", "纳米", "吨", "升", "克", "米", "秒", "度", "年", "天", "月", "周",
    ]

    @classmethod
    def _nums(cls, text: str):
        """D.parse_numbers 的架构层包装：先剥离单位词、屏蔽"裸单字中文数字"，再解析。
        修复冒烟发现的误判①："一个标准大气压"的"一"被当成 1，
        与锚点 100 比对 L=2.0 误触越界。
        修复全量发现的误判②："千米"的"千"被当成 1000（详见 _UNIT_WORDS 注释）。
        域本体文件（已公开）不动，此修正属于仂感头实现层。
        "三十万/一百/三十五"等组合表达不受影响。
        ★ 只在数值提取层做替换，不改动原文——触发词匹配（如"每秒"）仍作用于原始文本。"""
        masked = text
        for w in cls._UNIT_WORDS:
            masked = masked.replace(w, " ")
        masked = re.sub(r"(?<![零一二三四五六七八九十百千万两])[零一二三四五六七八九两](?![零一二三四五六七八九十百千万两])",
                        "□", masked)
        return D.parse_numbers(masked)

    # ---------- [1] 任务求存预检 ----------
    def scan_task(self, task: str):
        """任务先进仂域求存：扫出任务触及哪些锚点、任务自带数字落在哪个区。
        这一步在模型看到任何 token 之前完成——预挂门控的观察清单。
        ★ 归属靠触发词，不靠"数值等于真值"：污染场景下任务里的数字是
          错误值（101≠100），按等值判定会让域以为任务没触及该锚点，
          门控清单落空、污染直接放行（首跑实测到此坑，已修）。"""
        touched, warnings = [], []
        sents = [s for s in _SENT_SPLIT.split(task) if s.strip()]
        for a in D.ANCHORS:
            trig = TRIGGERS.get(a["id"], [])
            rel = [s for s in sents if any(t in s for t in trig)]
            hit = bool(rel) or any(s and s in task
                                   for s in [a["surface"]] + a.get("alt", []))
            if hit:
                touched.append(a["id"])
            if not rel:
                continue
            # 任务声明值 = 触发句里与真值偏离最大的数字（注入错误在此现形）
            cands = [a["value"]] + list(a.get("equiv", []))
            worst = None
            for v in self._nums("。".join(rel)):
                for cv in cands:
                    if v > 0 and cv > 0:
                        L = round(math.log10(max(v / cv, cv / v)), 4)
                        if worst is None or L > worst["L"]:
                            worst = {"value": v, "L": L}
            if worst and worst["L"] > 1e-9:
                warnings.append({"anchor": a["id"], "task_value": worst["value"],
                                 "L": worst["L"],
                                 "zone": D.classify(a["value"], worst["value"])["zone"]})
        return {"touched": touched, "warnings": warnings}

    # ---------- [3] 仂感头（在仂域里）----------
    def sense(self, text: str, anchor_ids, claimed=None, task_ctx=None):
        """仂感头：对生成文本按外置锚做数值级 L 判据。
        两道闸门（冒烟修正版）：
          闸1 语义归属：锚点是否被谈论，看「任务上下文 + 输出」的触发词；
                       但参与 L 比对的数字只取输出文本——归属看上下文，定数看输出。
                       （contam 口径下触发词在 prompt、输出只有纯数字，
                        只看输出会把真污染漏成"未涉及"）
          闸2 污染复述：输出任何位置精确出现任务声明值 claimed（≠真值时）
              → 判 CONTAMINATED（与 06 目录口径2 同规则，可比）。
        判定核 = leyu_number_domain 的 parse_numbers / classify，纯规则、不训练。"""
        claimed = claimed or {}
        results = []
        ctx = (f"{task_ctx}。" if task_ctx else "") + text
        sents = [s for s in _SENT_SPLIT.split(ctx) if s.strip()]
        for aid in anchor_ids:
            a = self.anchors[aid]
            wv = claimed.get(aid)                      # 任务声明值（注入错误在此）
            nums_all = self._nums(text)
            # 闸2：污染复述（精确相等，全文本扫）
            if wv is not None and any(abs(v - wv) / abs(wv) <= 1e-6 for v in nums_all):
                c = D.classify(a["value"], wv)
                results.append({"anchor": aid, "state": "CONTAMINATED",
                                "L": c["L"], "value": wv,
                                "zone": c["zone"], "via": "复述声明值"})
                continue
            # 闸1：触发词激活（同句数字才参与）
            trig = TRIGGERS.get(aid, [])
            rel = [s for s in sents if any(t in s for t in trig)]
            if not rel:
                results.append({"anchor": aid, "state": "MISS", "L": None,
                                "zone": "未涉及"})
                continue
            cands = [a["value"]] + list(a.get("equiv", []))
            best = None
            for v in self._nums(text):          # 定数只看输出
                for cv in cands:
                    if v > 0 and cv > 0:
                        L = math.log10(max(v / cv, cv / v))
                        if best is None or L < best["L"]:
                            best = {"value": v, "L": round(L, 4)}
            if best is None:
                results.append({"anchor": aid, "state": "MISS", "L": None,
                                "zone": "未涉及"})
                continue
            zone = ("中庸区" if best["L"] < 0.003 else
                    ("偏离区" if best["L"] < 1.0 else "越界区"))
            state = ("RECOVERED" if zone == "中庸区" else
                     ("CONTAMINATED" if zone == "越界区" else "DEVIATED"))
            results.append({"anchor": aid, "state": state, **best, "zone": zone,
                            "via": "触发句"})
        worst = max((r for r in results if r["L"] is not None),
                    key=lambda r: r["L"], default=None)
        return {"per_anchor": results, "worst": worst,
                "zone": worst["zone"] if worst else "无关",
                "hit_true": any(r["state"] == "RECOVERED" for r in results)}

    # ---------- [4] 门控（在仂域内）：仂感头判定 + 三态裁决 + 撤时域供给 ----------
    def supply(self, anchor_id):
        """域供给：撤时给出域内锚点正确值。"""
        a = self.anchors[anchor_id]
        return a["tpl"].replace("{n}", a["surface"])

    def _claimed(self, task: str, watch):
        """任务侧声明值：任务文本中触发词同句的数字（注入错误由此现形）。"""
        claimed = {}
        sents = [s for s in _SENT_SPLIT.split(task) if s.strip()]
        for aid in watch:
            a = self.anchors[aid]
            trig = TRIGGERS.get(aid, [])
            rel = [s for s in sents if any(t in s for t in trig)]
            if not rel:
                continue
            cands = [a["value"]] + list(a.get("equiv", []))
            # 任务声明值 = 触发句里与真值偏离最大的数字（注入错误的声明）
            worst = None
            for v in self._nums("。".join(rel)):
                for cv in cands:
                    if v > 0 and cv > 0:
                        L = math.log10(max(v / cv, cv / v))
                        if worst is None or L > worst["L"]:
                            worst = {"value": v, "L": round(L, 4)}
            if worst and worst["L"] > 1e-9:
                claimed[aid] = worst["value"]
        return claimed

    def adjudicate(self, text: str, task: str, watch, round_no: int = 1):
        """门控在仂域内完成：管线不参与判定，只执行本裁决。
        输入 = 任务全文 + 输出文本（门控看得见上下文，判定不出域）。
        行 = 放行；待 = 重问（管线拿 retry_hint 回主干）；撤 = 域供给正确值。
        重问后的第二轮输出必须再次送回本方法（round_no=2）。"""
        claimed = self._claimed(task, watch)
        s = self.sense(text, watch, claimed=claimed, task_ctx=task)
        zone = s["zone"]
        if zone in ("无关", "未涉及", "中庸区"):
            return {"action": "行", "final": text, "sense": s, "round": round_no,
                    "claimed": claimed}
        if zone == "偏离区":
            return {"action": "待", "final": text, "sense": s, "round": round_no,
                    "claimed": claimed,
                    "retry_hint": "以上说法中的数字是错的。请给出正确说法。"}
        # 越界区 → 撤：拦截并附域内锚点正确值
        bad = s["worst"]["anchor"]
        supplied = self.supply(bad)
        final = text + f"\n【仂域·撤】L={s['worst']['L']:.3f} 越界。域内锚点：{supplied}"
        s2 = self.sense(final, watch, claimed=claimed, task_ctx=task)
        return {"action": "撤", "final": final, "sense": s2, "sense_raw": s,
                "supply": {"anchor": bad, "text": supplied}, "round": round_no,
                "claimed": claimed}

    def record(self, entry):
        self.ledger.append(entry)


# ============================================================ [2][4] 管线
class GatedPipeline:
    """求存预检 → 干净主干 → 输出门控（仂感头在域内）。"""

    def __init__(self, tok, mdl, domain: LeyuDomain):
        self.tok, self.mdl, self.domain = tok, mdl, domain
        self.counts = {"行": 0, "待": 0, "撤": 0}

    @torch.no_grad()
    def _gen(self, prompt, max_new=MAX_NEW):
        """[2] 干净主干：分词 → 嵌入 → 神经网络 → 输出。无任何 hook/头。"""
        ids = self.tok(prompt, return_tensors="pt")           # 分词库
        # 嵌入层显式走一遍（架构陈述用；generate 内部同样路径）
        emb = self.mdl.get_input_embeddings()(ids["input_ids"])
        out = self.mdl.generate(input_ids=ids["input_ids"],
                                attention_mask=ids["attention_mask"],
                                max_new_tokens=max_new, do_sample=False,
                                pad_token_id=self.tok.eos_token_id)
        text = self.tok.decode(out[0][ids["input_ids"].shape[1]:],
                               skip_special_tokens=True)
        return text, emb.shape

    def generate(self, task):
        """完整门控管线：①求存 → ②③干净前向 → ④输出过仂域门控。
        管线自身无判定逻辑：门控裁决（行/待/撤）完全由仂域给出，
        重问后的第二轮输出再次送回仂域门控（两轮都经过仂域）。"""
        t0 = time.time()
        scan = self.domain.scan_task(task)                    # ① 任务进仂域求存
        watch = scan["touched"]
        text, emb_shape = self._gen(task)                     # ②③ 干净主干
        # ④ 门控（仂域内，看得见任务全文+输出全文）
        v1 = self.domain.adjudicate(text, task, watch, round_no=1)

        entry = {"task": task, "raw": text, "scan": scan,
                 "adjudication_round1": v1, "emb_shape": list(emb_shape)}

        if v1["action"] == "待":
            # 待：管线按裁决重问（回干净主干），第二轮输出再次过仂域门控
            text2, _ = self._gen(task + v1["retry_hint"])
            v2 = self.domain.adjudicate(text2, task, watch, round_no=2)
            entry["adjudication_round2"] = v2
            final_v = v2
        else:
            final_v = v1

        action = final_v["action"]
        self.counts[action] += 1
        entry.update({"action": action, "final": final_v["final"],
                      "final_sense": final_v["sense"], "rounds": final_v["round"],
                      "sec": round(time.time() - t0, 1)})
        self.domain.record(entry)
        return {"action": action, "final": final_v["final"],
                "final_sense": final_v["sense"], "scan": scan, "entry": entry}


# ============================================================ 实验入口
def build_prompts(case):
    """三口径 prompt，与 06_仂域_数字域_v1/leyu_probe_v1.py 逐字对齐：
      contam  = 错误陈述 + 原样重问（逼模型补完数字，才测得出是否被带跑）
      recover = 明确告知有错 + 重问
    第一版只喂陈述句，模型自由续写不谈数字 → 对照失真，已修正。"""
    stmt = case["statement"]
    probe = case["probe"]
    p = {
        "probe": probe,
        "contam": f"{stmt}{probe}",
        "recover": f"{stmt}这句话里的数字是错的。{probe}",
    }
    wrong_surf = None if case["error_type"] == "L0" else case["surface"]
    return p, wrong_surf


def judge_fixed(text, case, mode="strict"):
    """终态判定（数值级，与 rejudge_m 同规则：污染优先，无子串兜底）。
    ★ 必须用完整锚点（含 surface / alt / equiv）——第一版把这三样清空，
      "30万"、等价单位等合法表述被判 MISS，命中率被低估。

    两种口径（都要报，不许只报好看的）：
      strict = 与 06 目录完全同尺可比：终态只要还留着错误值就判 CONTAMINATED。
               撤的终态含"原文(错)+域供给(对)"，严格口径下仍是 CONTAMINATED。
      gated  = 架构口径：域已拦截该错误值并显式给出锚点正确值（含【仂域·撤】标记），
               对下游读者而言这就是被纠回，判 RECOVERED。
               除撤之外，两口径完全一致。"""
    anchor = next(a for a in D.ANCHORS if a["id"] == case["anchor_id"])
    ht = D.hit_test(text, anchor)
    if mode == "gated" and "【仂域·撤】" in text and ht["hit_true"]:
        return "RECOVERED"
    if case["error_type"] != "L0" and D.hit_wrong_test(text, case["value_given"]):
        return "CONTAMINATED"
    return "RECOVERED" if ht["hit_true"] else "MISS"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="4 例快测")
    ap.add_argument("--repeat", type=int, default=N_REPEAT)
    args = ap.parse_args()

    random.seed(SEED); torch.manual_seed(SEED)
    print("=" * 88)
    print("仂域求存门控架构 v1 ·  对照实验（无门控 baseline vs 门控管线）")
    print("=" * 88)
    print(f"  模型: {MODEL_NAME}  解码: greedy  种子: {SEED}  max_new: {MAX_NEW}")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16,
                                               trust_remote_code=True)
    mdl.eval()
    print("模型就绪。\n")

    domain = LeyuDomain()
    pipe = GatedPipeline(tok, mdl, domain)

    cases = D.build_cases()
    if args.smoke:
        cases = [c for c in cases if c["anchor_name"] in ("水的沸点", "圆周率")][:8]

    all_runs = []
    for rep in range(args.repeat):
        torch.manual_seed(SEED); random.seed(SEED)
        print(f"\n{'#'*88}\n# 第 {rep+1} / {args.repeat} 遍\n{'#'*88}")
        run = {"rep": rep, "records": []}
        for ci, case in enumerate(cases):
            prompts, wrong_surf = build_prompts(case)

            # ---- 一次生成，两路对照 ----
            # baseline 取自同一次生成的 raw 输出（门控前的原文），
            # 而不是"再单独生成一次"——fp16 CPU 多线程存在偶发数值非确定性，
            # 两次生成同 prompt 可能出不同文本（实测已见 1 例），那样对照就不严格了。
            t0 = time.time()
            g = pipe.generate(prompts["contam"])
            raw_text = g["entry"]["raw"]
            base_state = judge_fixed(raw_text, case, mode="strict")
            gated_strict = judge_fixed(g["final"], case, mode="strict")
            gated_eff = judge_fixed(g["final"], case, mode="gated")

            rec = {
                "anchor_id": case["anchor_id"], "anchor_name": case["anchor_name"],
                "error_type": case["error_type"], "surface": case["surface"],
                "value_true": case["value_true"], "value_given": case["value_given"],
                "L": case["domain_measured"]["L"],
                "baseline": {"text": raw_text, "state": base_state,
                             "sec": round(time.time() - t0, 1)},
                "gated": {"action": g["action"], "final": g["final"],
                          "state": gated_strict, "state_effective": gated_eff,
                          "sense_zone": g["final_sense"]["zone"],
                          "scan_touched": g["scan"]["touched"],
                          "scan_warnings": g["scan"]["warnings"]},
            }
            run["records"].append(rec)
            if (ci + 1) % 10 == 0 or args.smoke:
                print(f"  [{ci+1}/{len(cases)}] {case['anchor_name']} {case['error_type']}"
                      f"  base={base_state}  gated严格={gated_strict}"
                      f"  gated架构={gated_eff} ({g['action']})")
        all_runs.append(run)

    # ---------------- 汇总 ----------------
    print("\n" + "=" * 88)
    print("结果汇总（终态正确率：RECOVERED / 40）")
    print("=" * 88)
    summary = {}
    for et in ("L0", "D1", "D2", "M"):
        rs = [r for r in all_runs[0]["records"] if r["error_type"] == et]
        b = sum(r["baseline"]["state"] == "RECOVERED" for r in rs)
        gs = sum(r["gated"]["state"] == "RECOVERED" for r in rs)          # 严格口径
        ge = sum(r["gated"]["state_effective"] == "RECOVERED" for r in rs)  # 架构口径
        acts = {}
        for r in rs:
            acts[r["gated"]["action"]] = acts.get(r["gated"]["action"], 0) + 1
        summary[et] = {"baseline_recovered": b, "gated_recovered_strict": gs,
                       "gated_recovered_effective": ge,
                       "n": len(rs), "actions": acts}
        print(f"  {et:3s}  baseline {b}/{len(rs)} ({b/len(rs):.0%})"
              f"  →  gated严格 {gs}/{len(rs)} ({gs/len(rs):.0%})"
              f"  →  gated架构 {ge}/{len(rs)} ({ge/len(rs):.0%})   动作{acts}")

    acts_all = pipe.counts
    recs = all_runs[0]["records"]
    print(f"\n  门控动作合计: 行 {acts_all['行']}  待 {acts_all['待']}  撤 {acts_all['撤']}")
    print(f"  baseline → gated架构 提升例数: "
          f"{sum(r['gated']['state_effective']=='RECOVERED' and r['baseline']['state']!='RECOVERED' for r in recs)}")
    # 非确定性自检：每遍之间的 raw 是否一致（fp16 CPU 多线程偶发翻转）
    nd = 0
    for i in range(len(recs)):
        vals = {run["records"][i]["baseline"]["text"] for run in all_runs}
        if len(vals) > 1:
            nd += 1
    if nd:
        print(f"  ★ 非确定性：{nd}/{len(recs)} 例在三遍之间 raw 输出不一致"
              f"（fp16 CPU 多线程数值翻转，如实记录）")

    # 复现性自检
    diff = 0
    total = 0
    for i in range(len(all_runs[0]["records"])):
        key = ("baseline", "gated")
        for k in key:
            vals = {(run["records"][i][k]["text"] if k == "baseline"
                     else run["records"][i][k]["final"]) for run in all_runs}
            total += 1
            if len(vals) > 1:
                diff += 1
    print(f"  复现性: {total-diff}/{total} 逐位一致")

    out = {"meta": {"model": MODEL_NAME, "arch": "leyu_gated_pipeline_v1",
                    "max_new": MAX_NEW, "n_repeat": args.repeat, "seed": SEED,
                    "n_cases": len(cases)},
           "gate_counts": acts_all, "summary": summary,
           "ledger": domain.ledger, "runs": all_runs}
    p_json = os.path.join(HERE, "gated_v1_result.json")
    with open(p_json, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[saved] {p_json}\nDone.")


if __name__ == "__main__":
    main()
