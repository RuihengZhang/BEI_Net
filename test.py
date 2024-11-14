import argparse
import csv
import sys
import torch
import inspect
import os
import shutil
from datetime import datetime
from functools import partial
from torch.cuda.amp import autocast, GradScaler

import pylab

import albumentations as A
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from mmengine import Config
from tqdm import tqdm

import datasets
import method as model_lib
from method.mssim import ssim
from utils import constructor, ops, pt_utils, py_utils
from utils.data import get_data_from_txt, get_datasets_info_with_keys, read_binary_array, read_color_array
from utils.recorder import AvgMeter, CalTotalMetric, MsgLogger, TimeRecoder

def iou(prob, gt): 
    inter = torch.sum(gt * prob, dim=(1, 2, 3)) 
    union = gt.sum(dim=(1, 2, 3)) + prob.sum(dim=(1, 2, 3)) - inter 
    iou = inter / union
    return iou.mean()

class TeDataset(torch.utils.data.Dataset): 
    def __init__(self, root, shape):
        super().__init__()
        self.datasets = get_datasets_info_with_keys(dataset_infos=root, extra_keys=["mask", "t"])
        self.image_paths = self.datasets["image"]
        self.mask_paths = self.datasets["mask"]
        self.t_paths = self.datasets["t"]

        self.joint_trans = A.Compose([A.Resize(height=shape["h"], width=shape["w"]), A.Normalize()])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]
        t_path = self.t_paths[index]

        image = read_color_array(image_path)
        t = read_binary_array(t_path, to_normalize=True, thr=-1)

        transformed = self.joint_trans(image=image, mask=t)
        image = transformed["image"]
        t = transformed["mask"]

        image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        t_tensor = torch.from_numpy(t).unsqueeze(0)

        return dict(
            image=image_tensor,
            t=t_tensor,
            image_info=dict(mask_path=mask_path, mask_name=os.path.basename(mask_path)),
        )


@torch.no_grad() 
def eval_once(model, data_loader, save_path="", show_bar=True):
    model.eval()
    cal_total_seg_metrics = CalTotalMetric()

    bar_iter = enumerate(data_loader)
    if show_bar:
        bar_iter = tqdm(bar_iter, total=len(data_loader), leave=False, ncols=79)
    for batch_id, batch in bar_iter:
        images = batch["image"].cuda(non_blocking=True) 
        ts = batch["t"].cuda(non_blocking=True)
        logits = model(data=dict(image=images, t=ts))
        logits = logits[0]
        probs = logits.sigmoid().squeeze(1).cpu().detach().numpy() 

        for i, pred in enumerate(probs):
            mask_path = batch["image_info"]["mask_path"][i]
            mask_array = read_binary_array(mask_path, dtype=np.uint8) 
            mask_h, mask_w = mask_array.shape 

            pred = cv2.resize(pred, dsize=(mask_w, mask_h), interpolation=cv2.INTER_LINEAR)  

            if save_path:  
                pred_name = os.path.splitext(batch["image_info"]["mask_name"][i])[0] + ".png"
                ops.save_array_as_image(data_array=pred, save_name=pred_name, save_dir=save_path)

            pred = (pred * 255).astype(np.uint8)
            cal_total_seg_metrics.step(pred, mask_array, mask_path)
    return cal_total_seg_metrics.get_results()


def testing(model, msg_logger, cfg):
    msg_logger(name="log", msg="\n", show=False)

    csv_row = [cfg.exp_name]
    for te_data_name in cfg.data.test.name:
        te_data_path = datasets.__dict__[te_data_name]
        te_dataset = TeDataset(root=(te_data_name, te_data_path), shape=cfg.data.test.shape)
        te_loader = torch.utils.data.DataLoader(
            dataset=te_dataset,
            batch_size=cfg.args.batch_size,
            num_workers=cfg.args.num_workers,
            pin_memory=True,
        )
        print(f"Testing on {te_data_name} with {len(te_dataset)} samples")
        pred_save_path = os.path.join(cfg.path.save, te_data_name)
        seg_results = eval_once(model=model, save_path=pred_save_path, data_loader=te_loader, show_bar=cfg.show_bar)
        msg_logger(name="log", msg=f"Results on {te_data_path}:\n{seg_results}")

        csv_row.extend(list(seg_results.values()))

    
    with open(cfg.path.csv, encoding="utf-8", mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_row)


def loss_func(logits, seg_gts):   
    losses = []
    loss_str = []
    
    bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(input=logits, target=seg_gts, reduction="mean")
    losses.append(bce_loss)
    loss_str.append(f"bce:{bce_loss.item():.5f}")

    prob = logits.sigmoid()
    ssim_loss = 1 - ssim(prob, seg_gts)
    losses.append(ssim_loss)
    loss_str.append(f"ssim:{ssim_loss.item():.5f}")

    iou_loss = 1 - iou(prob, seg_gts)
    losses.append(iou_loss)
    loss_str.append(f"iou:{iou_loss.item():.5f}")
    return sum(losses), " ".join(loss_str)

def parse_config():
    parser = argparse.ArgumentParser() 
    parser.add_argument("--config", type=str, default='./configs/rgbt.py')
    parser.add_argument("--output-root", type=str, default="./output")
    parser.add_argument("--model-name", type=str, default='BEINet_R101')
    parser.add_argument("--load-from", type=str, default="")
    parser.add_argument("--pretrained", type=str, default="")
    parser.add_argument("--info", type=str, default='rgbt')
    parser.add_argument("--evaluate", action="store_true") 
    parser.add_argument("--show-bar", action="store_true")
    parser.add_argument("--cooldown-epoch-num", type=int, default=0)
    args = parser.parse_args()

    print(args.config[0])

    cfg = Config.fromfile(args.config, use_predefined_variables=False) 
    cfg.merge_from_dict(vars(args)) 
    cfg.exp_name = py_utils.construct_exp_name(config=cfg) 
    if cfg.cooldown_epoch_num > 0:
        cfg.exp_name += f"_CD{cfg.cooldown_epoch_num}"

    cfg.path = py_utils.construct_path(output_root=cfg.output_root, exp_name=cfg.exp_name)  
    cfg.vis_path = os.path.join(cfg.path.exp, "imgs")

    os.makedirs(cfg.path.exp, exist_ok=True)
    os.makedirs(cfg.path.save, exist_ok=True)
    os.makedirs(cfg.path.pth, exist_ok=True)

    with open(cfg.path.log, encoding="utf-8", mode="w") as f:
        f.write(f"=== {datetime.now()} ===\n")
    with open(cfg.path.cfg, encoding="utf-8", mode="w") as f:
        f.write(cfg.pretty_text)
    shutil.copy(__file__, cfg.path.trainer) 

    if os.path.exists(cfg.vis_path):
        shutil.rmtree(cfg.vis_path)
    os.makedirs(cfg.vis_path)

    metric_names = ["Smeasure", "wFmeasure", "MAE", "adpEm", "meanEm", "maxEm", "adpFm", "meanFm", "maxFm"]
    with open(cfg.path.csv, encoding="utf-8", mode="w", newline="") as f:
        writer = csv.writer(f)

        first_row = ["model_name"]
        for dataset_name in cfg.data.test.name:
            first_row.extend([dataset_name] + [" "] * (len(metric_names) - 1))
        writer.writerow(first_row)

        second_row = [" "] + metric_names * len(cfg.data.test.name)
        writer.writerow(second_row)
    return cfg


def main():
    cfg = parse_config()
    pt_utils.initialize_seed_cudnn(seed=cfg.args.base_seed, deterministic=cfg.args.deterministic)  
    print(f"[{datetime.now()}] {cfg.path.exp} with base_seed {cfg.args.base_seed}") 

    msg_logger = MsgLogger(log=cfg.path.log)

    if hasattr(model_lib, cfg.model_name):
        ModuleClass = getattr(model_lib, cfg.model_name)
        model = ModuleClass(pretrained=cfg.pretrained)   
        msg_logger(name="log", msg=inspect.getsource(ModuleClass)) 
    else:
        raise ModuleNotFoundError(f"Please add <{cfg.model_name}> into the __init__.py.")

    if cfg.load_from: 
        model.load_state_dict(torch.load(cfg.load_from, map_location="cpu"))
        print(f"Loaded from {cfg.load_from}")

    model.cuda()
    testing(model=model, msg_logger=msg_logger, cfg=cfg)

if __name__ == "__main__":
    main()
