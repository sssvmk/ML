#!/usr/bin/env python3
"""python run_pipeline.py PATH [cpu|gpu|combination]  -- the complete SSD pipeline, no other parameters."""
import sys

from ssd_voc.pipeline import main

if __name__ == "__main__":
    sys.exit(main())
