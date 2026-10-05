SMOKE = [
    "data.dataset=synthetic", "data.synthetic.size=40", "data.synthetic.image_size=96", "data.synthetic.classes=3",
    "data.num_workers=0", "data.batch_size=4", "data.train_eval_images=8",
    "model.backbone=resnet18", "model.pretrained=none", "model.freeze_bn=false", "model.hidden_dim=64", "model.nheads=4",
    "model.enc_layers=1", "model.dec_layers=2", "model.dim_feedforward=128", "model.num_queries=10",
    "aug.scales=[96,112]", "aug.max_size=160", "aug.crop_resize=[80,96]", "aug.crop_min=48", "aug.crop_max=80", "aug.test_size=96",
    'schedule.phases=[{"lr":0.0003,"epochs":2},{"lr":0.00003,"epochs":1}]', "schedule.eval_every=1", "schedule.log_every=2",
    "sanity.overfit_steps=60", "metric.target_value=0.01", "mlflow.experiment_name=smoke", "mlflow.backend=sqlite"]
