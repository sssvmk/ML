#!/usr/bin/env python3
"""
auto_reconcile.py - automatic reconciliation of SAP open items (BSEG + BKPF)

    source_data.SourceData  ->  .history     -> mine_history()  FP-Growth association rules (UNSUPERVISED)
    (company code + GL)     ->  .open_items  -> solve           CSP WITH BACKTRACKING (branch and bound)

Always enforced : 1-10 (zero net, items once, whole items, same company/account/partner/special GL/
                  currency, debit+credit, eligible items) and 16-18 (reversal pairs, tolerance, closed periods)
Optional flags  : 11 pattern allowed | 12 group size bounded | 13 date order | 14 date window | 15 references
                  (all default ON; switch off with Config(enforce_pattern=False, ...) or CLI --skip 11,14)

Run:  python auto_reconcile.py --demo --bukrs IN04 --hkont 0001400001
      python auto_reconcile.py --csv extract.csv --bukrs IN04 --hkont 0001400001 --cutoff 2022-12-31 --skip 14
Spark:
      src = SourceData.from_spark(spark, "IN04", "0001400001", date(2022, 12, 31))
      res = run(src, Config(closed_years=(2022,)))
"""
from __future__ import annotations

import argparse
import math
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd

from source_data import EPOCH, SourceData

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.setrecursionlimit(10000)

OPTIONAL = {11: "enforce_pattern", 12: "enforce_size", 13: "enforce_date_order",
            14: "enforce_date_window", 15: "enforce_references"}


# --------------------------------------------------------------------------- config
@dataclass
class Config:
    # ---- optional constraint flags (True = enforced) ----
    enforce_pattern: bool = True          # 11  mix of debit/credit types must exist in mined history
    enforce_size: bool = True             # 12  lines per group <= historical maximum for the pattern
    enforce_date_order: bool = True       # 13  clearing date >= every member's posting date
    enforce_date_window: bool = True      # 14  gap between members within window learned from history
    enforce_references: bool = True       # 15  invoice reference (REBZG) targets must be in the group
    clearing_date: date | None = None     # 13: proposed clearing date (default = cutoff)
    strict_refs: bool = False             # 15: additionally all non-blank ZUONR in a group must be equal
    hard_max_group_size: int = 8          # search bound used when 12 is off (keeps the search finite)
    # ---- mining ----
    require_rules: bool = True            # fail loudly if the unsupervised step (mlxtend) is unavailable
    min_pattern_count: int = 3
    min_pattern_support: float = 0.01
    rule_min_support: float = 0.02
    rule_min_confidence: float = 0.30
    date_slack_days: int = 7
    # ---- always-on settings ----
    tolerance_minor: dict = field(default_factory=dict)     # constraint 17, e.g. {"INR": 0}
    closed_years: tuple = ()                                # constraint 18
    # ---- solver ----
    prematch_pairs: bool = True
    top_k_groups: int = 4
    enum_cap: int = 5000
    node_budget: int = 3000
    time_budget_s: float = 15.0

    def flags(self) -> dict:
        return {k: getattr(self, v) for k, v in OPTIONAL.items()}


def confirm_algorithms(cfg: Config, src: SourceData) -> None:
    """Printed at the start of every run, and the run stops if the unsupervised step cannot work."""
    try:
        import mlxtend  # noqa: F401
        rules_ok = f"mlxtend {mlxtend.__version__}"
    except ImportError:
        if cfg.require_rules:
            raise ImportError("Unsupervised step needs mlxtend (pip install mlxtend), or set require_rules=False")
        rules_ok = "NOT AVAILABLE (rules skipped; require_rules=False)"
    print("=" * 78)
    print("ALGORITHM CONFIRMATION")
    print(f"  Scope           : company code {src.bukrs} | GL account {src.hkont} | cutoff {src.cutoff}")
    print(f"  Source data     : {src.summary()['history_lines']} cleared lines -> mining, "
          f"{src.summary()['open_lines']} open lines -> solver")
    print(f"  Unsupervised    : FP-Growth association rules, {rules_ok} (no labels, no target variable)")
    print("                    + frequency counting of whole debit/credit patterns")
    print("  Solver          : CSP with backtracking (branch and bound over candidate groups)")
    print("  Optional flags  : " + " | ".join(f"{k} {'ON' if v else 'OFF'}" for k, v in cfg.flags().items()))
    print("=" * 78)


# --------------------------------------------------------------------------- mining (unsupervised)
def mine_history(hist: pd.DataFrame, cfg: Config):
    """Returns (patterns, rules, stats) from cleared groups of one company code + one GL account."""
    keys = ["bukrs", "augbl", "auggj"]
    tol = lambda ccy: cfg.tolerance_minor.get(ccy, 0)
    net = hist.groupby(keys + ["waers"]).signed.sum().reset_index()
    net["ok"] = [abs(v) <= tol(c) for v, c in zip(net.signed, net.waers)]
    balanced = net.groupby(keys).ok.all()

    agg = hist.groupby(keys).agg(sigs=("sig", frozenset), n=("sig", "size"),
                                 partners=("partner", "nunique"), ccys=("waers", "nunique"),
                                 d0=("day", "min"), d1=("day", "max"))
    agg["gap"] = agg.d1 - agg.d0
    agg["balanced"] = balanced
    two_sided = agg.sigs.map(lambda s: any(x.startswith("S|") for x in s) and any(x.startswith("H|") for x in s))
    train = agg[agg.balanced & (agg.partners == 1) & (agg.ccys == 1) & two_sided]
    stats = dict(history_groups=len(agg), training_groups=len(train),
                 dropped_unbalanced=int((~agg.balanced).sum()),
                 dropped_multi_partner=int((agg.partners > 1).sum()),
                 dropped_multi_currency=int((agg.ccys > 1).sum()))
    if train.empty:
        raise ValueError(f"No usable cleared history groups: {stats}")

    pat = (train.groupby("sigs").agg(n_groups=("n", "size"), max_lines=("n", "max"), gap_max=("gap", "max"))
           .reset_index())
    pat["support"] = pat.n_groups / len(train)
    pat["allowed"] = (pat.n_groups >= cfg.min_pattern_count) & (pat.support >= cfg.min_pattern_support)
    pat = pat.sort_values("n_groups", ascending=False).reset_index(drop=True)
    return pat, mine_rules(train, cfg), stats


def mine_rules(train: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """FP-Growth association rules over item signatures."""
    try:
        from mlxtend.frequent_patterns import association_rules, fpgrowth
        from mlxtend.preprocessing import TransactionEncoder
    except ImportError:
        if cfg.require_rules:
            raise
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])
    try:
        baskets = [sorted(s) for s in train.sigs]
        te = TransactionEncoder()
        X = pd.DataFrame(te.fit(baskets).transform(baskets), columns=te.columns_)
        fi = fpgrowth(X, min_support=cfg.rule_min_support, use_colnames=True)
        try:
            r = association_rules(fi, metric="confidence", min_threshold=cfg.rule_min_confidence,
                                  num_itemsets=len(X))
        except TypeError:                                     # older mlxtend
            r = association_rules(fi, metric="confidence", min_threshold=cfg.rule_min_confidence)
        return r[["antecedents", "consequents", "support", "confidence", "lift"]] \
            .sort_values("lift", ascending=False).reset_index(drop=True)
    except Exception as e:                                    # e.g. no itemset with two items
        print(f"[rules] none produced: {e}")
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])


# --------------------------------------------------------------------------- constraint context
class Ctx:
    def __init__(self, patterns: pd.DataFrame, rules: pd.DataFrame, cfg: Config, cutoff: date):
        self.cfg = cfg
        allowed = patterns[patterns.allowed]
        self.allowed = {r.sigs: dict(max_lines=int(r.max_lines), gap_max=int(r.gap_max), support=float(r.support))
                        for r in allowed.itertuples()}
        if not self.allowed:
            raise ValueError("No pattern passes the support threshold; lower min_pattern_count/min_pattern_support")
        # global limits: fallback for patterns not in history when constraint 11 is off
        self.g = dict(max_lines=max(v["max_lines"] for v in self.allowed.values()),
                      gap_max=max(v["gap_max"] for v in self.allowed.values()),
                      support=min(v["support"] for v in self.allowed.values()))
        self.max_size = self.g["max_lines"] if cfg.enforce_size else cfg.hard_max_group_size
        self.clearing_day = (pd.Timestamp(cfg.clearing_date or cutoff) - EPOCH).days
        self._cache: dict = {}
        self.conf = {}
        for r in rules.itertuples():
            a, c = list(r.antecedents), list(r.consequents)
            if len(a) == 1 and len(c) == 1 and a[0].startswith("S|") and c[0].startswith("H|"):
                self.conf[(a[0], c[0])] = float(r.confidence)

    def subset_ok(self, sigs: frozenset) -> bool:
        """Pruning for constraint 11: a partial group must still fit inside some mined pattern."""
        if not self.cfg.enforce_pattern:
            return True
        v = self._cache.get(sigs)
        if v is None:
            v = any(sigs <= p for p in self.allowed)
            self._cache[sigs] = v
        return v

    def info_for(self, sigs: frozenset):
        info = self.allowed.get(sigs)
        if info is None and not self.cfg.enforce_pattern:
            info = self.g                                       # unseen mix: use global learned limits
        return info

    def eval_group(self, P, g) -> float | None:
        """Return a score if the group satisfies every active constraint, else None."""
        c = self.cfg
        sigs = frozenset(P.sig[i] for i in g)
        info = self.info_for(sigs)
        if info is None:                                                         # 11
            return None
        if c.enforce_size and len(g) > info["max_lines"]:                        # 12
            return None
        if abs(sum(P.amt[i] for i in g)) > P.tol:                                # 1, 17
            return None
        deb = [i for i in g if P.amt[i] > 0]
        cred = [i for i in g if P.amt[i] < 0]
        if not deb or not cred:                                                  # 9
            return None
        days = [P.day[i] for i in g]
        gap = max(days) - min(days)
        if c.enforce_date_window and gap > info["gap_max"] + c.date_slack_days:  # 14
            return None
        if c.enforce_date_order and max(days) > self.clearing_day:               # 13
            return None
        gs, docs = set(g), {P.doc[i] for i in g}
        for i in g:                                                              # 15 (+16 via must)
            if not P.must[i] <= gs or not P.req_docs[i] <= docs:
                return None
        if c.enforce_references and c.strict_refs and len({P.zuonr[i] for i in g if P.zuonr[i]}) > 1:
            return None
        link = ref_link(P, deb, cred, docs)
        pairs = [self.conf.get((P.sig[a], P.sig[b]), 0.0) for a in deb for b in cred]
        return math.log(info["support"]) + 2.0 * link + sum(pairs) / len(pairs) - 0.02 * gap - 0.05 * len(g)


def ref_link(P, deb, cred, docs) -> int:
    dt = {t for i in deb for t in P.tok[i]}
    ct = {t for i in cred for t in P.tok[i]}
    if dt & ct:
        return 1
    return int(any(P.req_docs[i] and P.req_docs[i] <= docs for i in deb + cred))


# --------------------------------------------------------------------------- CSP with backtracking
def enumerate_groups(P, ctx: Ctx, anchor: int, free: frozenset):
    """Valid groups containing `anchor` and only free items (top-k by score)."""
    base, seen, stack = [anchor], {anchor}, [anchor]
    while stack:                                   # forced members (reversal / single-target reference)
        for r in P.must[stack.pop()]:
            if r not in free:
                return []
            if r not in seen:
                seen.add(r), base.append(r), stack.append(r)
    sigs0 = frozenset(P.sig[i] for i in base)
    if not ctx.subset_ok(sigs0):
        return []
    cands = sorted((j for j in free if j not in seen), key=lambda j: -abs(P.amt[j]))
    m = len(cands)
    pos, neg = [0] * (m + 1), [0] * (m + 1)
    for k in range(m - 1, -1, -1):
        a = P.amt[cands[k]]
        pos[k], neg[k] = pos[k + 1] + max(a, 0), neg[k + 1] + min(a, 0)
    tol, out, cnt = P.tol, [], [0]

    def dfs(start, chosen, total, sigs):
        cnt[0] += 1
        if cnt[0] > ctx.cfg.enum_cap:
            return
        if abs(total) <= tol and (chosen or len(base) > 1):
            g = tuple(base + chosen)
            s = ctx.eval_group(P, g)
            if s is not None:
                out.append((s, g))
        if len(base) + len(chosen) >= ctx.max_size:
            return
        for k in range(start, m):
            if total + pos[k] < -tol or total + neg[k] > tol:    # bound: target interval unreachable
                break
            j = cands[k]
            ns = sigs if P.sig[j] in sigs else sigs | {P.sig[j]}
            if not ctx.subset_ok(ns):                             # prune: no mined pattern fits
                continue
            chosen.append(j)
            dfs(k + 1, chosen, total + P.amt[j], ns)
            chosen.pop()                                          # backtrack

    dfs(0, [], sum(P.amt[i] for i in base), sigs0)
    out.sort(key=lambda x: (-x[0], len(x[1])))
    return out[: ctx.cfg.top_k_groups]


def prematch_pairs(P, ctx: Ctx, free: set):
    """Cheap pass: exact 1:1 opposite amounts, best score first."""
    by_amt = defaultdict(list)
    for j in free:
        if P.amt[j] < 0:
            by_amt[-P.amt[j]].append(j)
    done = []
    for i in sorted((i for i in free if P.amt[i] > 0), key=lambda i: P.day[i]):
        scored = [(ctx.eval_group(P, (i, j)), j) for j in by_amt.get(P.amt[i], [])]
        scored = [(s, j) for s, j in scored if s is not None]
        if scored:
            s, j = max(scored)
            by_amt[P.amt[i]].remove(j)
            free -= {i, j}
            done.append(((i, j), s))
    return done


def solve_partition(P, ctx: Ctx):
    """Branch and bound with backtracking. Groups are enumerated by their first item in `order`;
    the other branch leaves that item open. Objective: most items cleared, then best total score."""
    n = len(P.amt)
    order = sorted(range(n), key=lambda i: -abs(P.amt[i]))
    best = {"cleared": -1, "score": -1e18, "groups": []}
    st = {"nodes": 0, "exhausted": False}
    deadline = time.monotonic() + ctx.cfg.time_budget_s

    def rec(p, free, groups, cleared, score):
        if st["nodes"] > ctx.cfg.node_budget or time.monotonic() > deadline:
            st["exhausted"] = True
            return
        st["nodes"] += 1
        while p < n and order[p] not in free:
            p += 1
        if p == n:
            if (cleared, score) > (best["cleared"], best["score"]):
                best.update(cleared=cleared, score=score, groups=list(groups))
            return
        if cleared + len(free) < best["cleared"]:                      # bound
            return
        anchor = order[p]
        for s, g in enumerate_groups(P, ctx, anchor, free):
            groups.append((g, s))
            rec(p + 1, free - frozenset(g), groups, cleared + len(g), score + s)   # constraint 2
            groups.pop()                                               # backtrack
            if st["exhausted"]:
                return
        rec(p + 1, free - {anchor}, groups, cleared, score)            # anchor stays open

    rec(0, frozenset(range(n)), [], 0, 0.0)
    return best["groups"], st["exhausted"]


# --------------------------------------------------------------------------- partitions
def _doc(bukrs, belnr, year, gjahr):
    return "|".join([bukrs, belnr, year or gjahr])


def build_partition(part: pd.DataFrame, cfg: Config) -> SimpleNamespace:
    part = part.reset_index(drop=True)
    n = len(part)
    P = SimpleNamespace(
        ids=part.item_id.tolist(), amt=part.signed.tolist(), sig=part.sig.tolist(), day=part.day.tolist(),
        doc=part.doc_key.tolist(), zuonr=part.zuonr.tolist(),
        tok=[frozenset(t for t in (("z", z), ("x", x)) if t[1]) for z, x in zip(part.zuonr, part.xblnr)],
        must=[set() for _ in range(n)], req_docs=[set() for _ in range(n)],
        tol=int(cfg.tolerance_minor.get(part.waers.iloc[0], 0)), df=part)
    by_doc = defaultdict(list)
    for i, dk in enumerate(P.doc):
        by_doc[dk].append(i)
    for i, r in enumerate(part.itertuples()):
        targets = []
        if cfg.enforce_references and r.rebzg:                    # 15: invoice reference
            targets.append(_doc(r.bukrs, r.rebzg, r.rebzj, r.gjahr))
        if r.stblg:                                               # 16: reversal counterpart (always on)
            targets.append(_doc(r.bukrs, r.stblg, r.stjah, r.gjahr))
        for t in targets:
            if t == r.doc_key or t not in by_doc:
                continue
            if len(by_doc[t]) == 1:
                P.must[i].add(by_doc[t][0])                       # single open line: forced into the group
            P.req_docs[i].add(t)                                  # group must hold >=1 line of that document
    return P


def reason_for(i, P, matched: set, exhausted: bool, doc_open: set) -> str:
    row = P.df.iloc[i]
    unmatched = [j for j in range(len(P.amt)) if j not in matched]
    opp = [j for j in unmatched if (P.amt[j] > 0) != (P.amt[i] > 0)]
    if not opp:
        return "ON_ACCOUNT_CREDIT_NO_OPEN_DEBIT" if P.amt[i] < 0 else "DEBIT_NO_OPEN_CREDIT"
    for j in opp:                                                 # partial-payment evidence via REBZG
        rj = P.df.iloc[j]
        small, big, rs = (row, rj, row) if abs(P.amt[i]) < abs(P.amt[j]) else (rj, row, rj)
        if rs.rebzg and _doc(rs.bukrs, rs.rebzg, rs.rebzj, rs.gjahr) == big.doc_key:
            return "PARTIAL_SUSPECTED"
    if row.rebzg and _doc(row.bukrs, row.rebzg, row.rebzj, row.gjahr) not in doc_open:
        return "REF_TARGET_NOT_OPEN"
    return "SEARCH_BUDGET_EXHAUSTED" if exhausted else "NO_VALID_ZERO_NET_COMBINATION"


# --------------------------------------------------------------------------- main run
def run(src: SourceData, cfg: Config) -> dict:
    confirm_algorithms(cfg, src)
    cutoff = src.cutoff
    patterns, rules, mstats = mine_history(src.history, cfg)         # unsupervised step
    ctx = Ctx(patterns, rules, cfg, cutoff)

    pool, review, groups = src.open_items, [], []
    for reason, mask in (("PARKED_OR_NOTED", pool.bstat != ""), ("ZERO_AMOUNT", pool.signed == 0)):
        done = {x["item_id"] for x in review}
        for r in pool[mask & ~pool.item_id.isin(done)].itertuples():
            review.append(dict(item_id=r.item_id, partner=r.partner, waers=r.waers, signed_minor=r.signed,
                               reason=reason))
    pool = pool[~pool.item_id.isin({x["item_id"] for x in review})]

    # --- 16: reversal pairs cleared together (pre-pass) ---
    used: set = set()
    idx = defaultdict(list)
    for r in pool.itertuples():
        idx[(r.doc_key, r.partner, r.waers, r.signed)].append(r.item_id)
    for r in pool[pool.stblg != ""].itertuples():
        if r.item_id in used:
            continue
        cp = _doc(r.bukrs, r.stblg, r.stjah, r.gjahr)
        for cand in idx.get((cp, r.partner, r.waers, -r.signed), []):
            if cand not in used:
                days = pool.loc[pool.item_id.isin([r.item_id, cand]), "day"]
                if cfg.enforce_date_order and days.max() > ctx.clearing_day:      # 13 applies to pairs too
                    for iid in (r.item_id, cand):
                        used.add(iid)
                        rr = pool[pool.item_id == iid].iloc[0]
                        review.append(dict(item_id=iid, partner=rr.partner, waers=rr.waers,
                                           signed_minor=rr.signed, reason="REVERSAL_POSTED_AFTER_CLEARING_DATE"))
                    break
                used |= {r.item_id, cand}
                groups.append(dict(method="REVERSAL_PAIR", items=[r.item_id, cand], score=0.0, confidence="HIGH"))
                break
    left = pool[~pool.item_id.isin(used)]
    no_cp = left[left.stblg != ""]                                    # 10: reversed doc without open counterpart
    for r in no_cp.itertuples():
        review.append(dict(item_id=r.item_id, partner=r.partner, waers=r.waers, signed_minor=r.signed,
                           reason="REVERSED_NO_OPEN_COUNTERPART"))
    work = left[~left.item_id.isin(set(no_cp.item_id))]
    doc_open = set(pool.doc_key)

    # --- 4-8: partition by partner + currency (company, account, special GL fixed by the scope) ---
    for (partner, ccy), part in work.groupby(["partner", "waers"]):
        P = build_partition(part, cfg)
        free = set(range(len(P.amt)))
        matched_groups = prematch_pairs(P, ctx, free) if cfg.prematch_pairs else []
        exhausted = False
        if free:
            sub = sorted(free)
            Q = build_partition(part.iloc[sub], cfg)
            g2, exhausted = solve_partition(Q, ctx)                   # CSP with backtracking
            matched_groups += [(tuple(sub[i] for i in g), s) for g, s in g2]
        matched = {i for g, _ in matched_groups for i in g}
        for g, s in matched_groups:
            deb = [i for i in g if P.amt[i] > 0]
            cred = [i for i in g if P.amt[i] < 0]
            link = ref_link(P, deb, cred, {P.doc[i] for i in g})
            groups.append(dict(method="PATTERN_CSP", items=[P.ids[i] for i in g], score=round(s, 3),
                               confidence="HIGH" if link else "LOW_AMOUNT_ONLY"))
        for i in range(len(P.amt)):
            if i not in matched:
                review.append(dict(item_id=P.ids[i], partner=partner, waers=ccy, signed_minor=P.amt[i],
                                   reason=reason_for(i, P, matched, exhausted, doc_open)))

    # --- assemble outputs ---
    by_id = pool.set_index("item_id", drop=False)
    closed = set(map(str, cfg.closed_years))
    rows = []
    for k, g in enumerate(sorted(groups, key=lambda x: -len(x["items"])), start=1):
        sub = by_id.loc[g["items"]]
        resid = int(sub.signed.sum())
        rows.append(dict(
            group_id=k, method=g["method"], confidence=g["confidence"], n_items=len(sub), score=g["score"],
            partner=sub.partner.iloc[0], waers=sub.waers.iloc[0], residual_minor=resid,
            pattern=" + ".join(sorted(set(sub.sig))),
            proposed_clearing_date=sub.budat.max().date(),
            action="PROPOSE_CLEARING",                                           # 18: proposal only
            clearing_period="CURRENT_OPEN_PERIOD" if closed & set(sub.gjahr) else "ANY_OPEN_PERIOD",
            adjustment_posting="CURRENT_OPEN_PERIOD" if resid != 0 else "",
            items=g["items"]))
    res = dict(groups=pd.DataFrame(rows), review=pd.DataFrame(review), patterns=patterns, rules=rules,
               mining_stats=mstats, pool=pool, all_items=src.items, ctx=ctx, cutoff=cutoff)
    res["validation"] = validate_groups(res, cfg)
    res["backtest"] = backtest(res)
    return res


# --------------------------------------------------------------------------- validation
def validate_groups(res: dict, cfg: Config) -> pd.DataFrame:
    """Independent re-check from the raw extract (does not reuse solver structures)."""
    g, pool, ctx, cutoff = res["groups"], res["pool"], res["ctx"], pd.Timestamp(res["cutoff"])
    by_id = pool.set_index("item_id", drop=False)
    doc_open = set(pool.doc_key)
    clr = pd.Timestamp(cfg.clearing_date or res["cutoff"])
    V, seen = defaultdict(int), defaultdict(int)
    for r in g.itertuples():
        sub = by_id.loc[[i for i in r.items if i in by_id.index]]
        is_rev = r.method == "REVERSAL_PAIR"
        for i in r.items:
            seen[i] += 1
        resid = int(sub.signed.sum())
        tol = cfg.tolerance_minor.get(sub.waers.iloc[0], 0)
        V[1] += abs(resid) > tol
        V[17] += abs(resid) > tol
        V[3] += len(sub) != len(r.items)
        V[4] += sub.bukrs.nunique() != 1
        V[5] += sub.hkont.nunique() != 1
        V[6] += sub.partner.nunique() != 1
        V[7] += sub.umskz.nunique() != 1
        V[8] += sub.waers.nunique() != 1
        V[9] += not ((sub.signed > 0).any() and (sub.signed < 0).any())
        V[10] += int((sub.budat > cutoff).any() or (sub.bstat != "").any()
                     or ((sub.stblg != "").any() and not is_rev))
        sigs = frozenset(sub.sig)
        info = ctx.allowed.get(sigs)
        if not is_rev:
            V[11] += info is None
            lim = info or ctx.g
            V[12] += len(sub) > lim["max_lines"]
            V[14] += (sub.day.max() - sub.day.min()) > lim["gap_max"] + cfg.date_slack_days
        V[13] += sub.budat.max() > clr
        docs = set(sub.doc_key)
        for x in sub.itertuples():
            if x.rebzg:
                t = _doc(x.bukrs, x.rebzg, x.rebzj, x.gjahr)
                V[15] += (t in doc_open and t != x.doc_key and t not in docs)
            if x.stblg:
                V[16] += _doc(x.bukrs, x.stblg, x.stjah, x.gjahr) not in docs
        V[18] += not (r.action == "PROPOSE_CLEARING" and (r.residual_minor == 0 or r.adjustment_posting == "CURRENT_OPEN_PERIOD"))
    V[2] = sum(1 for v in seen.values() if v > 1)

    names = {1: "Zero net per group", 2: "Each item used once", 3: "Whole items only", 4: "Same company code",
             5: "Same GL account", 6: "Same partner", 7: "Same special GL indicator", 8: "Same currency",
             9: "At least one debit and one credit", 10: "Eligible items only", 11: "Pattern allowed",
             12: "Group size bounded", 13: "Date order valid", 14: "Date window valid",
             15: "References consistent", 16: "Reversal pairs together", 17: "Tolerance bounded",
             18: "Closed periods respected (proposal only)"}
    rows = []
    for k in range(1, 19):
        on = cfg.flags().get(k, True)
        status = ("PASS" if V[k] == 0 else "FAIL") if on else "SKIPPED (flag off)"
        rows.append(dict(check=k, name=names[k], violations=int(V[k]), status=status))

    rv = res["review"]
    skipped = {"PARKED_OR_NOTED", "ZERO_AMOUNT", "REVERSED_NO_OPEN_COUNTERPART", "REVERSAL_POSTED_AFTER_CLEARING_DATE"}
    excl = set(rv[rv.reason.isin(skipped)].item_id) if len(rv) else set()
    unmatched = (set(rv.item_id) if len(rv) else set()) - excl
    universe = set(pool.item_id) - excl
    net = lambda ids: int(pool[pool.item_id.isin(ids)].signed.sum())
    resid_total = int(g.residual_minor.sum()) if len(g) else 0
    ok19 = net(universe) == net(unmatched) + resid_total
    rows.append(dict(check=19, name="Imbalance preserved (before = unmatched + group residuals)",
                     violations=int(not ok19), status="PASS" if ok19 else "FAIL"))
    rows.append(dict(check=20, name="Company code nets to zero", violations=0,
                     status="NOT CHECKABLE (single-account extract)"))
    ok21 = (set(seen) | unmatched) == universe and not (set(seen) & unmatched)
    rows.append(dict(check=21, name="Every eligible item in exactly one group or the review queue",
                     violations=int(not ok21), status="PASS" if ok21 else "FAIL"))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- backtest
def backtest(res: dict) -> dict:
    """Compare proposals with clearing SAP did after the cutoff (time-travel backtest)."""
    d, pool, g = res["all_items"], res["pool"], res["groups"]
    cut = pd.Timestamp(res["cutoff"])
    truth = d[(d.augbl != "") & (d.augdt > cut)]
    if truth.empty or g.empty:
        return {}
    in_pool = set(pool.item_id)
    tg = truth.groupby(["bukrs", "augbl", "auggj"]).item_id.apply(frozenset)
    evaluable = {s for s in tg if s <= in_pool}          # true groups fully visible at the cutoff
    prop = [frozenset(x) for x in g["items"]]
    exact = [p for p in prop if p in evaluable]
    return dict(true_groups_evaluable=len(evaluable), proposed_groups=len(prop), exact_matches=len(exact),
                precision=round(len(exact) / len(prop), 3) if prop else None,
                recall=round(len(exact) / len(evaluable), 3) if evaluable else None,
                items_correct=sum(len(p) for p in exact), items_evaluable=sum(len(s) for s in evaluable))


# --------------------------------------------------------------------------- report / CLI
def report(res: dict):
    pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_colwidth", 60)
    print("\n=== Mining stats ===");  print(res["mining_stats"])
    print("\n=== Patterns (top 8) ===")
    print(res["patterns"].head(8).assign(sigs=lambda x: x.sigs.map(lambda s: " + ".join(sorted(s)))))
    r = res["rules"].head(6).copy()
    print("\n=== Association rules (top 6 by lift) ===")
    if len(r):
        r["antecedents"] = r.antecedents.map(lambda s: ",".join(sorted(s)))
        r["consequents"] = r.consequents.map(lambda s: ",".join(sorted(s)))
    print(r)
    g, rv = res["groups"], res["review"]
    print(f"\n=== Proposed groups: {len(g)} | items cleared: {int(g.n_items.sum()) if len(g) else 0} "
          f"| review queue: {len(rv)} ===")
    if len(g):
        print(g.groupby(["method", "confidence"]).agg(groups=("group_id", "size"), items=("n_items", "sum")))
    if len(rv):
        print("\nReview queue by reason:"); print(rv.reason.value_counts())
    print("\n=== Constraint validation ===")
    print(res["validation"][["check", "name", "violations", "status"]].to_string(index=False))
    if res["backtest"]:
        print("\n=== Backtest vs what SAP actually cleared later ==="); print(res["backtest"])


def main():
    ap = argparse.ArgumentParser(description="SAP open-item auto reconciliation")
    ap.add_argument("--bukrs", required=True, help="company code, e.g. IN04")
    ap.add_argument("--hkont", required=True, help="GL account number, e.g. 0001400001")
    ap.add_argument("--cutoff", help="YYYY-MM-DD (default: 2022-12-31 for --demo, otherwise today)")
    ap.add_argument("--umskz", default="", help="special GL indicator (default: normal items)")
    ap.add_argument("--skip", default="", help="optional constraints to switch off, e.g. 11,12,13,14,15")
    ap.add_argument("--closed-years", default="", help="e.g. 2022,2021")
    ap.add_argument("--demo", action="store_true"), ap.add_argument("--csv")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()

    skip = {int(x) for x in a.skip.split(",") if x.strip()}
    if skip - set(OPTIONAL):
        ap.error(f"--skip accepts only {sorted(OPTIONAL)}; the other constraints are always enforced")
    cfg = Config(closed_years=tuple(y for y in a.closed_years.split(",") if y),
                 **{OPTIONAL[k]: False for k in skip})
    cutoff = date.fromisoformat(a.cutoff) if a.cutoff else (date(2022, 12, 31) if a.demo else date.today())
    if a.demo:
        src = SourceData.from_demo(a.bukrs, a.hkont, cutoff, a.umskz)
    elif a.csv:
        src = SourceData.from_csv(a.csv, a.bukrs, a.hkont, cutoff, a.umskz)
    else:
        ap.error("give --demo or --csv (or call SourceData.from_spark from a notebook)")
    res = run(src, cfg)
    report(res)
    res["groups"].assign(items=lambda x: x["items"].map(";".join)).to_csv(f"{a.out}/proposed_groups.csv", index=False)
    res["review"].to_csv(f"{a.out}/review_queue.csv", index=False)
    res["validation"].to_csv(f"{a.out}/validation.csv", index=False)


if __name__ == "__main__":
    main()
