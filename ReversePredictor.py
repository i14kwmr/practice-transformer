import torch.nn.functional as F

from TransformerPredictor import TransformerPredictor


# F: pytorchで実装されてる関数
# pred.viewについて(https://qiita.com/kenta1984/items/d68b72214ce92beebbe2)
# pred.size(-1)は最後の次元を持ってくる． 最後の次元は恐らく系列長となってるはず
class ReversePredictor(TransformerPredictor):  # テンプレートTransformerPredictor(pl.LightningModule)を継承
    def _calculate_loss(self, batch, mode="train"):
        # Fetch data and transform categories to one-hot vectors
        inp_data, labels = batch
        inp_data = F.one_hot(inp_data, num_classes=self.hparams.num_classes).float()  # one hotベクトルに変換

        # Perform prediction and calculate loss and accuracy
        preds = self.forward(inp_data, add_positional_encoding=True)  # PEありで学習
        loss = F.cross_entropy(preds.view(-1, preds.size(-1)), labels.view(-1))  # この時点でpredsは次元が少し違う（はず）．predsのshapeを(preds.size(-1), ...)に変更
        acc = (preds.argmax(dim=-1) == labels).float().mean()  # predsでは確率のようなものが入っているため最大値を取っている．

        # Logging
        self.log("%s_loss" % mode, loss)
        self.log("%s_acc" % mode, acc)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, _ = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="val")  # validationではlossは必要なし

    def test_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="test")  # testではlossは必要なし