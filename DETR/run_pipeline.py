#!/usr/bin/env python3
"""python run_pipeline.py PATH [cpu|gpu]  -- load datasets, train DETR, test once, save the model."""
import sys

from detr_voc.pipeline import main

if __name__ == "__main__":
    sys.exit(main())
