"""
source_data.py - the single data source for the reconciliation pipeline.

The user supplies a company code and a GL account number. SourceData loads BSEG + BKPF lines for
that scope and produces two views of the same prepared data:

    .history     lines already cleared at the cutoff   -> feeds association-rule mining (unsupervised)
    .open_items  lines open at the cutoff              -> feeds the CSP with backtracking solver
    .items       everything in scope (used by the backtest)

Constructors: from_spark(), from_csv(), from_frame(), from_demo().
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

CCY_DECIMALS = {"JPY": 0, "KRW": 0, "VND": 0, "CLP": 0, "ISK": 0,
                "KWD": 3, "BHD": 3, "OMR": 3, "JOD": 3, "TND": 3}
EPOCH = pd.Timestamp("1970-01-01")
TEXT_COLS = ["bukrs", "belnr", "gjahr", "buzei", "hkont", "kunnr", "lifnr", "umskz", "shkzg", "bschl",
             "blart", "waers", "zuonr", "xblnr", "rebzg", "rebzj", "stblg", "stjah", "augbl", "auggj", "bstat"]


def acct(a) -> str:
    """GL account with SAP leading zeros (numeric accounts are padded to 10 characters)."""
    a = str(a).strip()
    return a.zfill(10) if a.isdigit() else a


def _prepare(raw: pd.DataFrame, bukrs: str, hkont: str, umskz: str) -> pd.DataFrame:
    d = raw.copy()
    d.columns = [c.lower() for c in d.columns]
    for c in TEXT_COLS:
        if c not in d:
            d[c] = ""
        d[c] = d[c].fillna("").astype(str).str.strip()
    for c in ("budat", "augdt"):
        s = d[c].astype(str).replace({"00000000": "", "None": "", "NaT": "", "nan": ""})
        try:                                   # mixed formats (e.g. 20221231 and 2022-12-31) per element
            d[c] = pd.to_datetime(s, errors="coerce", format="mixed")
        except ValueError:                     # pandas < 2
            d[c] = pd.to_datetime(s, errors="coerce")
    d["hkont"] = d["hkont"].map(acct)
    d = d[(d.bukrs == bukrs) & (d.hkont == hkont) & (d.umskz == umskz)]
    bad = int(d["budat"].isna().sum())
    if bad:
        print(f"[source_data] WARNING: {bad} line(s) dropped because the posting date (BUDAT) could not be read")
    d = d.dropna(subset=["budat"]).copy()
    if d.empty:
        raise ValueError(f"No lines found for company code {bukrs}, GL account {hkont}, special GL '{umskz}'")

    dec = d["waers"].map(lambda c: CCY_DECIMALS.get(c, 2)).astype(float)
    d["amt"] = np.rint(d["wrbtr"].astype(float).abs() * np.power(10.0, dec)).astype(np.int64)
    d["signed"] = np.where(d["shkzg"] == "S", d["amt"], -d["amt"])       # debit +, credit -
    d["sig"] = d["shkzg"] + "|" + d["blart"] + "|" + d["bschl"]          # item signature
    d["partner"] = np.where(d["kunnr"] != "", d["kunnr"], d["lifnr"])
    d["item_id"] = d["bukrs"] + "-" + d["belnr"] + "-" + d["gjahr"] + "-" + d["buzei"]
    d["doc_key"] = d["bukrs"] + "|" + d["belnr"] + "|" + d["gjahr"]
    d["day"] = (d["budat"] - EPOCH).dt.days.astype(int)
    return d.reset_index(drop=True)


@dataclass
class SourceData:
    bukrs: str
    hkont: str
    cutoff: date
    umskz: str = ""
    items: pd.DataFrame = None
    history: pd.DataFrame = None
    open_items: pd.DataFrame = None

    # ---- constructors ------------------------------------------------------------------------
    @classmethod
    def from_frame(cls, raw: pd.DataFrame, bukrs: str, hkont: str, cutoff: date, umskz: str = ""):
        if not str(bukrs).strip() or not str(hkont).strip():
            raise ValueError("company code and GL account number are required")
        s = cls(str(bukrs).strip(), acct(hkont), cutoff, umskz)
        s.items = _prepare(raw, s.bukrs, s.hkont, s.umskz)
        cut = pd.Timestamp(cutoff)
        cleared = (s.items.augbl != "") & (s.items.augdt <= cut)         # cleared as of the cutoff
        s.history = s.items[cleared]
        s.open_items = s.items[(s.items.budat <= cut) & ~cleared]        # constraint 10: open + posted
        if s.history.empty:
            raise ValueError("No cleared history at the cutoff, so there is nothing to mine rules from")
        return s

    @classmethod
    def from_spark(cls, spark, bukrs: str, hkont: str, cutoff: date, umskz: str = "",
                   bseg="t_erp_r2r_rbp_nonconf.bseg", bkpf="t_erp_r2r_rbp_conf.bkpf"):
        q = f"""
        select s.bukrs, s.belnr, s.gjahr, s.buzei, s.hkont, s.kunnr, s.lifnr, s.umskz,
               s.shkzg, s.bschl, s.wrbtr, s.dmbtr, s.zuonr, s.rebzg, s.rebzj,
               s.augbl, s.auggj, s.augdt,
               k.blart, k.waers, k.budat, k.xblnr, k.stblg, k.stjah, k.bstat
        from {bseg} s join {bkpf} k
          on s.bukrs = k.bukrs and s.belnr = k.belnr and s.gjahr = k.gjahr
        where s.bukrs = '{bukrs}' and s.hkont = '{acct(hkont)}' and s.gjahr <= '{cutoff.year}'
        """
        return cls.from_frame(spark.sql(q).toPandas(), bukrs, hkont, cutoff, umskz)

    @classmethod
    def from_csv(cls, path: str, bukrs: str, hkont: str, cutoff: date, umskz: str = ""):
        return cls.from_frame(pd.read_csv(path, dtype=str), bukrs, hkont, cutoff, umskz)

    @classmethod
    def from_demo(cls, bukrs: str, hkont: str, cutoff: date, umskz: str = "", seed: int = 11, customers: int = 25):
        return cls.from_frame(make_demo(bukrs, hkont, seed, customers), bukrs, hkont, cutoff, umskz)

    # ---- info --------------------------------------------------------------------------------
    def summary(self) -> dict:
        return dict(company_code=self.bukrs, gl_account=self.hkont, cutoff=str(self.cutoff),
                    lines_in_scope=len(self.items), history_lines=len(self.history),
                    open_lines=len(self.open_items))


# --------------------------------------------------------------------------- synthetic data
def make_demo(bukrs: str, hkont: str, seed: int = 11, customers: int = 25) -> pd.DataFrame:
    """Synthetic customer-account history: pay, credit memo, split payments, multi-invoice, reversal,
    plus open invoices, partial payments and on-account payments that must stay unmatched."""
    rnd = random.Random(seed)
    rows, ctr = [], {"b": 1900000000, "a": 1500000000}

    def nb():
        ctr["b"] += 1
        return str(ctr["b"]).zfill(10)

    def na():
        ctr["a"] += 1
        return str(ctr["a"]).zfill(10)

    def line(cust, belnr, d, shkzg, bschl, blart, amt, zuonr="", xblnr="", rebzg="", stblg="", stjah="",
             aug="", augdt=None):
        rows.append(dict(bukrs=bukrs, belnr=belnr, gjahr=str(d.year), buzei="001", hkont=acct(hkont),
                         kunnr=cust, lifnr="", umskz="", shkzg=shkzg, bschl=bschl, wrbtr=amt / 100,
                         dmbtr=amt / 100, zuonr=zuonr, rebzg=rebzg, rebzj=str(d.year) if rebzg else "",
                         augbl=aug, auggj=str(augdt.year) if aug else "", augdt=augdt, blart=blart,
                         waers="INR", budat=d, xblnr=xblnr, stblg=stblg, stjah=stjah, bstat=""))

    kinds = ["pay", "cm_pay", "split", "multi", "reversal", "open_inv", "partial", "onacct"]
    weights = [45, 12, 8, 12, 3, 8, 6, 6]
    lag = lambda: rnd.choice([0, 0, 1, 2, 5, 10, 30, 60])
    for c in range(customers):
        cust = f"C{1000 + c}"
        for _ in range(rnd.randint(18, 30)):
            d0 = date(2021, 1, 1) + timedelta(days=rnd.randint(0, 900))
            A = rnd.randint(50_000, 5_000_000)
            ref = f"INV{rnd.randint(10**6, 10**7)}"
            inv, kind = nb(), rnd.choices(kinds, weights)[0]
            dt = lambda lo, hi: d0 + timedelta(days=rnd.randint(lo, hi))
            zp = lambda: ref if rnd.random() < 0.7 else ""
            if kind == "pay":
                pdt, pay, aug = dt(15, 60), nb(), na()
                ad = pdt + timedelta(days=lag())
                line(cust, inv, d0, "S", "01", "DR", A, ref, ref, aug=aug, augdt=ad)
                line(cust, pay, pdt, "H", "15", "DZ", A, zp(), "BNK" + pay[-5:], aug=aug, augdt=ad)
            elif kind == "cm_pay":
                cm, pay, aug = nb(), nb(), na()
                x = int(A * rnd.uniform(0.05, 0.3))
                cdt, pdt = dt(5, 30), dt(30, 70)
                ad = pdt + timedelta(days=lag())
                line(cust, inv, d0, "S", "01", "DR", A, ref, ref, aug=aug, augdt=ad)
                line(cust, cm, cdt, "H", "11", "DG", x, ref, "CM" + cm[-5:], rebzg=inv if rnd.random() < .8 else "",
                     aug=aug, augdt=ad)
                line(cust, pay, pdt, "H", "15", "DZ", A - x, zp(), "BNK" + pay[-5:], aug=aug, augdt=ad)
            elif kind == "split":
                p1 = int(A * rnd.uniform(0.3, 0.7))
                a1, a2, aug = nb(), nb(), na()
                d1, d2 = dt(10, 30), dt(31, 60)
                ad = d2 + timedelta(days=lag())
                line(cust, inv, d0, "S", "01", "DR", A, ref, ref, aug=aug, augdt=ad)
                line(cust, a1, d1, "H", "15", "DZ", p1, zp(), "BNK" + a1[-5:], aug=aug, augdt=ad)
                line(cust, a2, d2, "H", "15", "DZ", A - p1, zp(), "BNK" + a2[-5:], aug=aug, augdt=ad)
            elif kind == "multi":
                B, inv2, pay, aug = rnd.randint(50_000, 3_000_000), nb(), nb(), na()
                ref2 = f"INV{rnd.randint(10**6, 10**7)}"
                pdt = dt(20, 50)
                ad = pdt + timedelta(days=lag())
                line(cust, inv, d0, "S", "01", "DR", A, ref, ref, aug=aug, augdt=ad)
                line(cust, inv2, d0 + timedelta(days=rnd.randint(0, 10)), "S", "01", "DR", B, ref2, ref2,
                     aug=aug, augdt=ad)
                line(cust, pay, pdt, "H", "15", "DZ", A + B, zp(), "BNK" + pay[-5:], aug=aug, augdt=ad)
            elif kind == "reversal":
                rv, aug = nb(), na()
                rdt = dt(0, 5)
                ad = rdt + timedelta(days=lag())
                line(cust, inv, d0, "S", "01", "DR", A, ref, ref, stblg=rv, stjah=str(rdt.year), aug=aug, augdt=ad)
                line(cust, rv, rdt, "H", "11", "DR", A, ref, ref, stblg=inv, stjah=str(d0.year), aug=aug, augdt=ad)
            elif kind == "open_inv":
                line(cust, inv, d0, "S", "01", "DR", A, ref, ref)
            elif kind == "partial":
                pay = nb()
                line(cust, inv, d0, "S", "01", "DR", A, ref, ref)
                line(cust, pay, dt(15, 50), "H", "15", "DZ", int(A * 0.6), ref, "BNK" + pay[-5:], rebzg=inv)
            else:
                pay = nb()
                line(cust, pay, d0, "H", "15", "DZ", A, "", "BNK" + pay[-5:])
    return pd.DataFrame(rows)
