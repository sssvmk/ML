#!/usr/bin/env python3
"""Single steps:  python run.py {list-stages|prepare-data|benchmark|train|test|package|predict} --dest PATH [--stage voc|coco] ...
The complete pipeline:  python run_pipeline.py PATH [cpu|gpu]"""
import sys

from detr_voc.cli import main

if __name__ == "__main__":
    sys.exit(main())
