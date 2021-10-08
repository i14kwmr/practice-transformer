import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

from CosineWarmupScheduler import CosineWarmupScheduler
from PositionalEncoding import PositionalEncoding
from TransformerEncoder import TransformerEncoder


class TransformerPredictor(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        model_dim,
        num_classes,
        num_heads,
        num_layers,
        lr,
        warmup,
        max_iters,
        dropout=0.0,
        input_dropout=0.0,
    ):
        """
        Args:
            input_dim: Hidden dimensionality of the input
            model_dim: Hidden dimensionality to use inside the Transformer, Transformerにおける隠れ次元（？）
            num_classes: Number of classes to predict per sequence element, 分類するクラス数（？）
            num_heads: Number of heads to use in the Multi-Head Attention blocks, 線形層でどれくらいhead作るか
            num_layers: Number of encoder blocks to use., encoderブロック数
            lr: Learning rate in the optimizer, 学習率
            warmup: Number of warmup steps. Usually between 50 and 500, ウォームアップステップの数, 上では100回とか？
            max_iters: Number of maximum iterations the model is trained for. This is needed for the CosineWarmup scheduler, モデルが学習される最大反復回数
            dropout: Dropout to apply inside the model, モデル内部で適用するドロップアウト
            input_dropout: Dropout to apply on the input features, 入力特徴に適用されるドロップアウト
        """
        super().__init__()
        self.save_hyperparameters()  # 引数をself.hparamsの属性にする（？）
        self._create_model()

    def _create_model(self):
        # Input dim -> Model dim
        self.input_net = nn.Sequential(
            nn.Dropout(self.hparams.input_dropout), nn.Linear(self.hparams.input_dim, self.hparams.model_dim)
        )
        # ---↑Input Embedding?--- #

        # Positional encoding for sequences
        self.positional_encoding = PositionalEncoding(d_model=self.hparams.model_dim)
        # ---↑Positional encoding?--- #

        # Transformer
        self.transformer = TransformerEncoder(
            num_layers=self.hparams.num_layers,
            input_dim=self.hparams.model_dim,
            dim_feedforward=2 * self.hparams.model_dim,
            num_heads=self.hparams.num_heads,
            dropout=self.hparams.dropout,
        )
        # ---↑Encorder and Decorder?--- #

        # Output classifier per sequence lement
        self.output_net = nn.Sequential(
            nn.Linear(self.hparams.model_dim, self.hparams.model_dim),
            nn.LayerNorm(self.hparams.model_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.model_dim, self.hparams.num_classes),
        )
        # ---↑Linear(Last)?--- #

    def forward(self, x, mask=None, add_positional_encoding=True):
        """
        Args:
            x: Input features of shape [Batch, SeqLen, input_dim]
            mask: Mask to apply on the attention outputs (optional)
            add_positional_encoding: If True, we add the positional encoding to the input.
                                      Might not be desired for some tasks.
        """
        x = self.input_net(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        x = self.transformer(x, mask=mask)
        x = self.output_net(x)
        return x

    @torch.no_grad()
    def get_attention_maps(self, x, mask=None, add_positional_encoding=True):
        """Function for extracting the attention matrices of the whole Transformer for a single batch.

        Input arguments same as the forward pass.
        """
        x = self.input_net(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        attention_maps = self.transformer.get_attention_maps(x, mask=mask)
        return attention_maps

# ここまではOK

# ここからはよくわからない（使われていないため）

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)

        # We don't return the lr scheduler because we need to apply it per iteration, not per epoch
        self.lr_scheduler = CosineWarmupScheduler(
            optimizer, warmup=self.hparams.warmup, max_iters=self.hparams.max_iters
        )
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler.step()  # Step per iteration

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError