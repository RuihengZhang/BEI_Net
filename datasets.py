_RGBT_COD_ROOT = "./data"


# RGB-T SOD  COD

train = dict(
    image=dict(path=f"{_RGBT_COD_ROOT}/train/RGB", suffix=".jpg"),
    t=dict(path=f"{_RGBT_COD_ROOT}/train/T", suffix=".png"),
    mask=dict(path=f"{_RGBT_COD_ROOT}/train/GT", suffix=".png"),
    edge=dict(path=f"{_RGBT_COD_ROOT}/train/edge", suffix=".png"),
)
test = dict(
    image=dict(path=f"{_RGBT_COD_ROOT}/test/RGB", suffix=".jpg"),
    t=dict(path=f"{_RGBT_COD_ROOT}/test/T", suffix=".png"),
    mask=dict(path=f"{_RGBT_COD_ROOT}/test/GT", suffix=".png"),
    edge=dict(path=f"{_RGBT_COD_ROOT}/train/edge", suffix=".png"),
)
                       