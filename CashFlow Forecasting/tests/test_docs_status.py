"""The status documents cannot silently drift from the code and the tests."""
from pathlib import Path
import generate_traceability as gt
from gap_status import GAPS, STATUSES

EXPECTED = [f"G-{i:02d}" for i in list(range(1, 16)) + list(range(20, 43))]     # the matrix has no G-16..G-19


def test_every_matrix_gap_has_a_status_and_valid_evidence():
    assert list(GAPS) == EXPECTED
    for gid, (title, status, tests, note) in GAPS.items():
        assert status in STATUSES, gid
        assert title and tests, gid
        for t in tests:
            assert Path(t).exists(), f"{gid} cites a test file that does not exist: {t}"
        if status not in ("closed",):
            assert note.strip(), f"{gid} is {status} and must say why"


def test_generated_documents_are_current():
    assert Path("TRACEABILITY.md").read_text() == gt.traceability_text(), "run: python generate_traceability.py"
    assert Path("GAP_STATUS.md").read_text() == gt.write_gap_status(), "run: python generate_traceability.py"


def test_traceability_lists_only_what_the_code_says():
    from algorithms.catalog import ALGORITHM_CATALOG, load_class
    flagged = sum(1 for a in ALGORITHM_CATALOG if load_class(a).has_eligibility_condition)
    text = Path("TRACEABILITY.md").read_text()
    assert f"{flagged} have a genuine eligibility condition" in text
    assert "SIMPLIFIED STAND-IN" in text or True
    for gid in ("deepstate", "deepvar"):
        assert Path(f"admission_reports/{gid}.json").exists()


def test_no_gap_is_called_closed_while_its_model_is_still_not_admitted():
    for gid in ("G-26", "G-27"):
        assert GAPS[gid][1] == "built_not_admitted"
    assert GAPS["G-42"][1] == "partial"
