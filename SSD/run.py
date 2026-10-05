#!/usr/bin/env python3
"""SSD (Liu et al., ECCV 2016) with a ResNet base network on Pascal VOC 2012 / COCO.

  python run.py list-stages
  python run.py prepare-data --dest /Volumes/main/default/ssd --stage voc
  python run.py benchmark    --dest ... --stage voc
  python run.py train        --dest ... --stage voc            # stage 2: detection training on VOC2012
  python run.py train        --dest ... --stage coco           # stage 3a: optional second round, COCO first
  python run.py train        --dest ... --stage voc_from_coco  # stage 3b: ... then fine-tune on VOC2012
  python run.py train        --dest ... --stage voc_long       # stage 4: longer schedule + zoom-out augmentation
  python run.py test         --dest ... --stage voc --run-id RUN      # VOC2007 test, once
  python run.py register     --dest ... --stage voc --run-id RUN
  python run.py predict      --dest ... --checkpoint .../checkpoints/voc_best.pt --image a.jpg
Stage 1 (ImageNet pretraining) is replaced by torchvision's pretrained ResNet, cached under --dest.
"""
import sys

from ssd_voc.cli import main

if __name__ == "__main__":
    sys.exit(main())
