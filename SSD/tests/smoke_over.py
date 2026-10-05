SMOKE = [
    "data.dataset=synthetic", "data.synthetic.size=40", "data.synthetic.image_size=128", "data.synthetic.classes=3",
    "data.num_workers=0", "data.batch_size=4", "data.train_eval_images=8",
    "model.backbone=resnet18", "model.pretrained=none", "model.freeze_bn=false", "model.input_size=128",
    "model.extras=[[128,256,2,1],[64,128,2,1]]", "model.boxes_per_loc=[4,6,6,4]",
    'schedule.phases=[{"lr":0.01,"iters":12},{"lr":0.001,"iters":6}]', "schedule.warmup_iters=3", "schedule.eval_every=6",
    "schedule.log_every=3", "schedule.ema.tau=5", "sanity.overfit_steps=40", "metric.target_value=0.01",
    "mlflow.experiment_name=smoke"]
