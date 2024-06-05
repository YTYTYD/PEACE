from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pickle
import torch
# from model.iTransformer import *
from transformers import AdamW
import torch.nn.functional as F
from mamba_ssm import Mamba

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
torch.cuda.manual_seed_all(1234)


# 定义数据集
class Dataset_CSV(Dataset):
    def __init__(self, train=True, fold=0, regression=1):

        # 打开文件用于二进制读取
        with open('./regression_' + str(regression) + '.pkl', 'rb') as file:
            all_feature_list, all_label_list, all_attention_list = pickle.load(file)
        if fold == 0:  # 用于五折选择训练集和测试集
            test_size = len(all_feature_list) // 5
            if train:
                self.feature_list = all_feature_list[:test_size * 4]
                self.label_list = all_label_list[:test_size * 4]
                self.attention_list = all_attention_list[:test_size * 4]
            else:
                self.feature_list = all_feature_list[test_size * 4:]
                self.label_list = all_label_list[test_size * 4:]
                self.attention_list = all_attention_list[test_size * 4:]
        if fold == 1:
            test_size = len(all_feature_list) // 5
            if train:
                self.feature_list = all_feature_list[:test_size] + all_feature_list[test_size * 2:]
                self.label_list = all_label_list[:test_size] + all_label_list[test_size * 2:]
                self.attention_list = all_attention_list[:test_size] + all_attention_list[test_size * 2:]
            else:
                self.feature_list = all_feature_list[test_size:test_size * 2]
                self.label_list = all_label_list[test_size:test_size * 2]
                self.attention_list = all_attention_list[test_size:test_size * 2]
        if fold == 2:
            test_size = len(all_feature_list) // 5
            if train:
                self.feature_list = all_feature_list[:test_size * 2] + all_feature_list[test_size * 3:]
                self.label_list = all_label_list[:test_size * 2] + all_label_list[test_size * 3:]
                self.attention_list = all_attention_list[:test_size * 2] + all_attention_list[test_size * 3:]
            else:
                self.feature_list = all_feature_list[test_size * 2:test_size * 3]
                self.label_list = all_label_list[test_size * 2:test_size * 3]
                self.attention_list = all_attention_list[test_size * 2:test_size * 3]
        if fold == 3:
            test_size = len(all_feature_list) // 5
            if train:
                self.feature_list = all_feature_list[:test_size * 3] + all_feature_list[test_size * 4:]
                self.label_list = all_label_list[:test_size * 3] + all_label_list[test_size * 4:]
                self.attention_list = all_attention_list[:test_size * 3] + all_attention_list[test_size * 4:]
            else:
                self.feature_list = all_feature_list[test_size * 3:test_size * 4]
                self.label_list = all_label_list[test_size * 3:test_size * 4]
                self.attention_list = all_attention_list[test_size * 3:test_size * 4]
        if fold == 4:
            test_size = len(all_feature_list) // 5
            if train:
                self.feature_list = all_feature_list[test_size:]
                self.label_list = all_label_list[test_size:]
                self.attention_list = all_attention_list[test_size:]
            else:
                self.feature_list = all_feature_list[:test_size]
                self.label_list = all_label_list[:test_size]
                self.attention_list = all_attention_list[:test_size]

    def __len__(self):
        return len(self.feature_list)

    def __getitem__(self, i):
        feature = self.feature_list[i]
        label = self.label_list[i]
        attention = self.attention_list[i]

        return feature, label, attention


def collate_fn(data):
    features = [i[0] for i in data]
    labels = [i[1] for i in data]
    attentions = [i[2] for i in data]

    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels)
    attentions = torch.LongTensor(attentions)

    return features, labels, attentions


# 假设的配置类，用于存储初始化模型所需的参数
# class Config:
#     def __init__(self):
#         self.seq_len = 47 # 每个患者最长有47条记录，即token
#         self.pred_len = 47
#         self.d_model = 41 # 每个token有41个特征
#         self.output_attention = False
#         self.use_norm = True
#         self.n_heads = 2
#         self.e_layers = 2
#         self.d_ff = 512
#         self.factor = 5
#         self.dropout = 0.2
#         self.activation = 'relu'
#         self.embed = 'fixed'  # 假设的嵌入类型
#         self.freq = 'h'  # 假设的频率类型
#         self.class_strategy = 'last'  # 假设的分类策略


class Network(nn.Module):
    def __init__(self, num_labels):
        super(Network, self).__init__()
        # self.configs = Config()
        self.dim = 41
        self.model = Mamba(d_model=self.dim, d_state=16, d_conv=4, expand=2, )

        self.classifier = nn.Linear(self.dim, 64)
        self.classifier2 = nn.Linear(64, num_labels)

    def forward(self, input):
        # itransformer
        outputs = self.model(input)
        # 分类
        logits_1 = self.classifier(outputs)

        logits_1 = F.relu(logits_1)
        logits_2 = self.classifier2(logits_1)

        return logits_2


# 函数用于计算 DataLoader 上的准确率
def evaluate_mse_mae(data_loader, model, device):
    model.eval()  # 将模型设置为评估模式
    total_mse_loss = []
    total_mae_loss = []
    total_valid_tokens = 0

    with torch.no_grad():  # 在评估阶段不计算梯度
        for features, labels, attentions in data_loader:
            features, labels, attentions = features.to(device), labels.to(device), attentions.to(device)
            outputs = model(features)

            # 将out和labels调整到相同的形状
            outputs = outputs.squeeze(-1)

            valid_out = outputs * attentions
            valid_labels = labels * attentions

            mse_loss = F.mse_loss(valid_out, valid_labels, reduction='none')
            mae_loss = F.l1_loss(valid_out, valid_labels, reduction='none')

            masked_mse_loss = mse_loss * attentions
            masked_mae_loss = mae_loss * attentions

            total_mse_loss.append(masked_mse_loss.sum())
            total_mae_loss.append(masked_mae_loss.sum())
            total_valid_tokens += attentions.sum()

    model.train()  # 将模型重新设置为训练模式
    return sum(total_mse_loss)/total_valid_tokens, sum(total_mae_loss)/total_valid_tokens


if __name__ == '__main__':
    dataset1 = Dataset_CSV()
    loader1 = DataLoader(dataset=dataset1,
                         batch_size=32,
                         collate_fn=collate_fn,
                         shuffle=True)

    # 创建测试集 DataLoader
    dataset_test = Dataset_CSV(train=False)
    loader_test = DataLoader(dataset=dataset_test, batch_size=32, collate_fn=collate_fn, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_network = Network(num_labels=1).to(device) # 回归
    # 训练
    optimizer = AdamW(model_network.parameters(), lr=5e-4)
    model_network.train()

    epochs = 50
    for epoch in range(epochs):
        total_mse_loss = []
        total_mae_loss = []
        total_valid_tokens = 0
        for i, (features, labels, attentions) in enumerate(loader1):
            features = features.to(device)
            labels = labels.to(device)
            attention_mask = attentions.to(device)

            out = model_network(features)

            # 将out和labels调整到相同的形状
            out = out.squeeze(-1)

            valid_out = out * attention_mask
            valid_labels = labels * attention_mask

            mse_loss = F.mse_loss(valid_out, valid_labels, reduction='none')
            mae_loss = F.l1_loss(valid_out, valid_labels, reduction='none')

            masked_mse_loss = mse_loss * attention_mask
            masked_mae_loss = mae_loss * attention_mask

            # 计算平均损失，只考虑有效的tokens
            mean_mse_loss = masked_mse_loss.sum() / attention_mask.sum()
            mean_mae_loss = masked_mae_loss.sum() / attention_mask.sum()

            mean_mse_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_mse_loss.append(masked_mse_loss.sum())
            total_mae_loss.append(masked_mae_loss.sum())
            total_valid_tokens += attention_mask.sum()

        print(f'训练MSE: {sum(total_mse_loss)/total_valid_tokens}，MAE:{sum(total_mae_loss)/total_valid_tokens}')

        test_mse, test_mae = evaluate_mse_mae(loader_test, model_network, device)

        print(f'测试MSE: {test_mse}，MAE:{test_mae}')