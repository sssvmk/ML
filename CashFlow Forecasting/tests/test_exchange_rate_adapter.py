"""Tests for ExchangeRateAdapter — the dedicated adapter for the Lai et al. 2018 FX benchmark."""
import pandas as pd
import pytest
from adapters.exchange_rate import ExchangeRateAdapter, CURRENCIES, DATASET_LABEL, DEFAULT_START
from contract import validate_contract, CONTRACT_COLUMNS, SeriesRole
from pipeline import Pipeline
from config import load_config

FILE = "/mnt/user-data/uploads/exchange_rate.txt"
pytestmark = pytest.mark.skipif(
    not __import__("pathlib").Path(FILE).exists(),
    reason="exchange_rate.txt not uploaded"
)


@pytest.fixture(scope="module")
def df():
    return ExchangeRateAdapter().extract(FILE)


# ---- file shape and provenance -----------------------------------------------
def test_produces_8_segments_one_per_currency(df):
    assert sorted(df["segment_id"].unique()) == sorted(f"FX-{c}" for c in CURRENCIES)


def test_7588_rows_per_segment(df):
    for sid, g in df.groupby("segment_id"):
        assert len(g) == 7588, f"{sid}: expected 7588 rows, got {len(g)}"


def test_date_range_matches_the_paper(df):
    assert df["date"].min() == pd.Timestamp(DEFAULT_START)
    assert df["date"].max() == pd.Timestamp("2010-10-10")


def test_all_contract_columns_are_present(df):
    assert set(CONTRACT_COLUMNS) <= set(df.columns)


def test_lineage_is_fully_populated(df):
    lin = df["lineage"].iloc[0]
    assert lin["source_adapter"] == "exchange_rate"
    assert lin["rule_version"] == "exchange-rate-v1"
    assert "exchange_rate.txt" in lin["extraction_batch_id"]


def test_dataset_label_is_fx_rate_not_a_cashflow_label(df):
    assert (df["dataset"] == DATASET_LABEL).all()
    assert DATASET_LABEL == "FX_Rate"
    from contract import Dataset
    assert DATASET_LABEL not in {d.value for d in Dataset if d.name != "GENERIC"}


def test_currency_field_carries_the_iso_code(df):
    for code in CURRENCIES:
        seg = df[df["segment_id"] == f"FX-{code}"]
        assert (seg["currency"] == code).all()


# ---- contract validation passes clean ----------------------------------------
def test_passes_contract_validation_with_no_errors_or_warnings(df):
    r = validate_contract(df)
    assert r.ok and not r.errors and not r.warnings


def test_all_values_are_numeric_and_positive(df):
    import numpy as np
    vals = df["value"].to_numpy()
    assert not np.isnan(vals).any()
    assert (vals > 0).all()


def test_all_rows_are_endogenous_no_exogenous(df):
    assert (df["series_role"] == SeriesRole.ENDOGENOUS.value).all()


def test_dates_are_strictly_monotone_per_segment(df):
    for sid, g in df.groupby("segment_id"):
        dates = g["date"].sort_values().reset_index(drop=True)
        assert dates.is_monotonic_increasing and not dates.duplicated().any()


# ---- subset extraction -------------------------------------------------------
def test_currency_subset_extracts_only_requested_currencies():
    df2 = ExchangeRateAdapter().extract(FILE, currencies=["AUD", "JPY"])
    assert sorted(df2["segment_id"].unique()) == ["FX-AUD", "FX-JPY"]
    assert validate_contract(df2).ok


def test_single_currency_extraction():
    df1 = ExchangeRateAdapter().extract(FILE, currencies=["GBP"])
    assert list(df1["segment_id"].unique()) == ["FX-GBP"]
    assert len(df1) == 7588


def test_custom_segment_prefix():
    df = ExchangeRateAdapter().extract(FILE, currencies=["AUD"], segment_prefix="BENCH-")
    assert list(df["segment_id"].unique()) == ["BENCH-AUD"]


def test_currency_order_matches_the_file_column_order():
    df = ExchangeRateAdapter().extract(FILE)
    # AUD is column 0: first date's value from pandas == first row's first column
    raw_aud_first = float(open(FILE).readline().split(",")[0])
    adapter_aud_first = float(df[df["segment_id"] == "FX-AUD"].sort_values("date")["value"].iloc[0])
    assert abs(adapter_aud_first - raw_aud_first) < 1e-6


# ---- error handling ----------------------------------------------------------
def test_missing_file_raises_file_not_found():
    with pytest.raises(FileNotFoundError, match="no_such_file"):
        ExchangeRateAdapter().extract("/tmp/no_such_file.txt")


def test_unknown_currency_code_raises_value_error():
    with pytest.raises(ValueError, match="Unknown currency codes"):
        ExchangeRateAdapter().extract(FILE, currencies=["USD"])


# ---- config-driven via Pipeline ----------------------------------------------
def test_pipeline_uses_exchange_rate_adapter_from_config(tmp_path):
    cfg = load_config("config.json")
    cfg["data_source"] = {"adapter": "exchange_rate",
                          "params": {"file_path": FILE, "currencies": ["GBP", "CHF"]}}
    pipe = Pipeline(cfg, tmp_path / "logs")
    segs = pipe.extract()
    v = pipe.validate(segs)
    assert set(segs) == {"FX-GBP", "FX-CHF"}
    assert all(r.ok for r in v.values())


def test_fx_data_has_dataset_attribute_not_direction():
    """The system knows only opaque grouping keys (company_code, currency, dataset).
    There is no direction concept anywhere downstream of the adapter."""
    from pooling import assemble_pool, PooledBatch
    df = ExchangeRateAdapter().extract(FILE, currencies=["GBP", "JPY"])
    segs = {sid: df[df["segment_id"]==sid] for sid in df["segment_id"].unique()}
    batch = assemble_pool(segs)
    cats = batch.categories()
    assert set(cats.keys()) == {"company_code", "currency", "dataset"}   # no direction key
    assert "direction" not in list(batch.attributes.values())[0]
    # dataset label is the adapter's opaque label — system treats it as a grouping key, nothing more
    assert all(a["dataset"] == "FX_Rate" for a in batch.attributes.values())
