
import torch
import torch.utils.data as data


# 入力: 大きさseq_lenの配列inp_data, 出力（ラベル）: 逆順のinp_data
# sizeはデータ数
class ReverseDataset(data.Dataset):
    def __init__(self, num_categories, seq_len, size):
        super().__init__()
        self.num_categories = num_categories
        self.seq_len = seq_len
        self.size = size

        self.data = torch.randint(self.num_categories, size=(self.size, self.seq_len))  # num_categories未満の整数

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        inp_data = self.data[idx]  # 入力データ
        labels = torch.flip(inp_data, dims=(0,))  # sizeの次元に沿って入れ替え
        return inp_data, labels