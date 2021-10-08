# Standard libraries
import math
import os
import urllib.request
from functools import partial
from urllib.error import HTTPError

# Plotting
# import matplotlib
# import matplotlib.pyplot as plt
import numpy as np
# PyTorch Lightning
import pytorch_lightning as pl
#import seaborn as sns
# PyTorch
import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
import torch.utils.data as data
# import torchmetrics
# Torchvision
# import torchvision
# from IPython.display import set_matplotlib_formats
from pytorch_lightning.callbacks import ModelCheckpoint

from ReverseDataset import ReverseDataset
from ReversePredictor import ReversePredictor

# from torchvision import transforms
# from torchvision.datasets import CIFAR100
# from tqdm.notebook import tqdm



def train_reverse(**kwargs):

    # Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
    DATASET_PATH = os.environ.get("PATH_DATASETS", "data/")
    # Path to the folder where the pretrained models are saved
    CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models/Transformers/")

    # Setting the seed
    pl.seed_everything(42)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", device)

    # Create a PyTorch Lightning trainer with the generation callback
    root_dir = os.path.join(CHECKPOINT_PATH, "ReverseTask")  # pathの作成
    os.makedirs(root_dir, exist_ok=True)

    # ---ここから---
    trainer = pl.Trainer(
        default_root_dir=root_dir,
        callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
        gpus=1 if str(device).startswith("cuda") else 0,
        max_epochs=10,
        gradient_clip_val=5,  # 追加のパラメータ
        progress_bar_refresh_rate=1,
    )
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "ReverseTask.ckpt")  # 事前学習しているかどうか
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = ReversePredictor.load_from_checkpoint(pretrained_filename)  # ロード
    else:
        model = ReversePredictor(max_iters=trainer.max_epochs * len(train_loader), **kwargs)  # 訓練
        trainer.fit(model, train_loader, val_loader)  # 学習ループ実行

    # Test best model on validation and test set
    val_result = trainer.test(model, test_dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, test_dataloaders=test_loader, verbose=False)
    result = {"test_acc": test_result[0]["test_acc"], "val_acc": val_result[0]["test_acc"]}

    model = model.to(device)
    return model, result


if __name__ == "__main__":
    dataset = partial(ReverseDataset, 10, 16)  # 0~9の整数を16個並べた配列，　残る引数はデータ数

    # loader作成
    train_loader = data.DataLoader(dataset(50000), batch_size=128, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = data.DataLoader(dataset(1000), batch_size=128)
    test_loader = data.DataLoader(dataset(10000), batch_size=128)

    # 学習とテスト
    reverse_model, reverse_result = train_reverse(
        input_dim=train_loader.dataset.num_categories,
        model_dim=32,
        num_heads=1,
        num_classes=train_loader.dataset.num_categories,
        num_layers=1,
        dropout=0.0,
        lr=5e-4,
        warmup=50,
    )

    print("Val accuracy:  %4.2f%%" % (100.0 * reverse_result["val_acc"]))
    print("Test accuracy: %4.2f%%" % (100.0 * reverse_result["test_acc"]))
