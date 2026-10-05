# TODO: what only you can do

## What was run and what was not
- Run (sandbox: 1 CPU core, no GPU, synthetic data and fake VOC/COCO trees): 71 tests pass. A 50-epoch synthetic run showed the training loss falling from 10.8 to 5.5, but its validation mAP stayed between 0 and 0.11 on only 4 validation images, so **no detection accuracy has been demonstrated**; DETR is slow to converge by design.
- NOT run: real VOC / COCO downloads and training, the ImageNet ResNet-50 weight download, anything on the A100 (bf16, speed, memory at batch 16 and 800x1333), Databricks workspace tracking. No mAP exists for real data.

## Do first
- [ ] `python run.py benchmark --dest PATH` (or step 1 of the pipeline): read `bottleneck`, `peak_gpu_mem_gb` and `est_hours_if_run_to_the_end`; reduce `data.batch_size` if the GPU runs out of memory, adjust `data.num_workers`.
- [ ] Confirm `metric.target_value` (0.50 is ASSUMED).
- [ ] Decide the dataset: VOC2012 is small for DETR; consider `--stage coco` (the paper's setting, about 18 GB, days of GPU time) or training on VOC2007 trainval + VOC2012 (`data.voc_train_sets`).
- [ ] If the validation mAP stays near zero for tens of epochs, read `next_steps.json` before waiting longer.

## Decisions for you
- [ ] A COCO-pretrained DETR fine-tuned on VOC (not implemented here) is the standard route for small datasets; say if you want it.
- [ ] No `combination` device mode: this pipeline has a single dataset stage, so there is nothing for the CPU to prepare while the GPU trains (the CPU already runs augmentation and the Hungarian matching).
- [ ] Not built: Databricks notebook version (the SSD builder can generate one), serving API, drift monitor, promotion gates, Deformable DETR.
- [ ] Keep the single file named `detr_pipeline.py`: the saved pyfunc imports it by that name.

## Assumptions
- ResNet-50 (ImageNet V1 weights), 100 queries, the paper's 300-epoch schedule for VOC; validation = 10% of VOC2012 trainval; test = VOC2007 test, used once; primary metric all-point AP@0.5 (VOC2012 protocol).
