_base_ = ["base.py"]

args = dict(
    base_seed=112358,
    batch_size=4,
    print_freq=20,
    epoch_num=150,
    # epoch_num=1,
    use_amp=True,
    iter_num=21840,
    epoch_based=True,
)

optimizers = dict(
    lr=0.005,
    strategy="all",
    optimizer="sgd",
    optimizer_candidates=dict(
        sgd=dict(
            momentum=0.9,
            weight_decay=5e-4,
            nesterov=False,
        ),
    ),
)

schedulers = dict(
    sche_usebatch=True,
    strategy="cos",
    scheduler_candidates=dict(
        cos=dict(
            warmup_length=1,
            min_coef=0.001,
            max_coef=1,
        ),
    ),
)

data = dict(
    train=dict(
        name=[
            "train",
        ],
        shape=dict(h=256, w=256),
    ),
    test=dict(
        name=[
            "test",
        ],
        shape=dict(h=256, w=256),
    ),
)
