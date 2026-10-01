"""One-time generator for the 12 deep-learning-family algorithm modules
that have no eligibility condition per PRD §3.2 (rows 20-22, 30-38).
Each produced file is a real, independent module file (per §6's mandatory
independence requirement) -- this script is dev tooling, not part of the
runtime package, and is kept for traceability of how the files were made.
"""
from pathlib import Path

PLAIN_ALGORITHMS = [
    ("rnn", "RNNModule", "RNN", 20, "trains on any signal, including noise, without mechanical failure"),
    ("lstm", "LSTMModule", "LSTM", 21, "trains on any signal, including noise, without mechanical failure"),
    ("gru", "GRUModule", "GRU", 22, "trains on any signal, including noise, without mechanical failure"),
    ("timesnet", "TimesNetModule", "TimesNet", 31, "no data-shape requirement"),
    ("itransformer", "ITransformerModule", "iTransformer", 32, "no data-shape requirement"),
    ("informer", "InformerModule", "Informer", 33, "no data-shape requirement"),
    ("autoformer", "AutoformerModule", "Autoformer", 34, "no data-shape requirement"),
    ("fedformer", "FEDformerModule", "FEDformer", 35, "no data-shape requirement"),
    ("etsformer", "ETSformerModule", "ETSformer", 36, "no data-shape requirement"),
    ("nbeats", "NBEATSModule", "N-BEATS", 37, "no data-shape requirement"),
    ("nhits", "NHiTSModule", "N-HiTS", 38, "no data-shape requirement"),
]

TEMPLATE = '''from __future__ import annotations
from ._neural_template import NeuralLagModule


class {cls}(NeuralLagModule):
    """
    {display} (PRD \\u00a73.2 #{num}). No eligibility condition -- {reason}.
    Uses the shared simplified feed-forward stand-in backbone
    (_neural_template.NeuralLagModule); see that module's docstring for
    why -- this is NOT a faithful {display} implementation.
    """

    name = "{name}"
    has_eligibility_condition = False
'''

if __name__ == "__main__":
    out_dir = Path(__file__).parent
    for fname, cls, display, num, reason in PLAIN_ALGORITHMS:
        content = TEMPLATE.format(cls=cls, display=display, num=num, reason=reason, name=fname)
        (out_dir / f"{fname}.py").write_text(content)
        print(f"wrote {fname}.py")
