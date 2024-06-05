from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pickle
import torch
# from model.iTransformer import *
from transformers import AdamW
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score
from sklearn.preprocessing import label_binarize
import numpy as np
from mamba_ssm import Mamba

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
torch.cuda.manual_seed_all(1234)


# 定义数据集
class Dataset_CSV(Dataset):
    def __init__(self, train=True, fold=0):

        # 打开文件用于二进制读取
        with open('./classify.pkl', 'rb') as file:
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
    labels = torch.LongTensor(labels)
    attentions = torch.LongTensor(attentions)

    return features, labels, attentions


# 假设的配置类，用于存储初始化模型所需的参数
# class Config:
#     def __init__(self):
#         self.seq_len = 47 # 每个患者最长有47条记录，即token
#         self.pred_len = 47 # 每个token有41个特征
#         self.d_model = 41
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

        self.classifier = nn.Linear(self.dim, num_labels)

    def forward(self, input):
        # itransformer
        outputs = self.model(input)

        # 分类
        logits = self.classifier(outputs)

        return logits


# 函数用于计算 DataLoader 上的准确率
def evaluate_metrics(data_loader, model, device, num_classes):
    model.eval()  # 将模型设置为评估模式
    total_correct = 0
    total_count = 0

    all_labels = []
    all_predictions = []
    with torch.no_grad():  # 在评估阶段不计算梯度
        for features, labels, attentions in data_loader:
            features, labels, attentions = features.to(device), labels.to(device), attentions.to(device)
            outputs = model(features)
            _, predictions = outputs.max(dim=2)

            valid_indices = attentions == 1
            valid_labels = labels[valid_indices]
            valid_predictions = predictions[valid_indices]

            all_labels.extend(valid_labels.cpu().numpy())
            all_predictions.extend(valid_predictions.cpu().numpy())

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    # 计算各种指标
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)

    # AUC计算，适用于二分类或将标签转换为one-hot编码的多分类
    if num_classes == 2:
        auc = roc_auc_score(all_labels, all_predictions)
    else:
        all_labels_one_hot = label_binarize(all_labels, classes=range(num_classes))
        all_predictions_one_hot = label_binarize(all_predictions, classes=range(num_classes))
        auc = roc_auc_score(all_labels_one_hot, all_predictions_one_hot, average='macro', multi_class='ovr')

    model.train()  # 将模型重新设置为训练模式

    return accuracy, auc, f1, recall, precision


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
    model_network = Network(num_labels=5).to(device)
    # 训练
    optimizer = AdamW(model_network.parameters(), lr=5e-4)
    model_network.train()

    epochs = 50
    for epoch in range(epochs):
        total_count = 0
        total_correct = 0
        for i, (features, labels, attentions) in enumerate(loader1):
            features = features.to(device)
            labels = labels.to(device)
            attention_mask = attentions.to(device)

            out = model_network(features)

            # 5 表示类别数量
            loss = F.cross_entropy(out.view(-1, 5), labels.view(-1), reduction='none')
            # 第一个参数表示batch_size, 第二个参数表示最大token长度
            loss = loss.view(out.shape[0], out.shape[1])  # 重新reshape成原始的形状
            loss = loss * attention_mask  # 只计算有效tokens的损失
            loss = loss.sum() / attention_mask.sum()  # 计算平均损失
            # loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # 计算准确的分类数
            _, predictions = out.max(dim=2)  # 获取预测的类别
            correct = (predictions == labels) & (attention_mask == 1)
            accuracy_count = correct.sum().item()  # 正确分类的个数
            total_valid_tokens = attention_mask.sum().item()  # 有效token数

            total_count += total_valid_tokens
            total_correct += accuracy_count

        print(f'训练准确率为：{total_correct / total_count}')
        test_accuracy, test_auc, test_f1, test_recall, test_precision = evaluate_metrics(loader_test, model_network,
                                                                                         device, num_classes=5)
        print(f'测试准确率为：{test_accuracy}, auc:{test_auc}, f1:{test_f1},recall:{test_recall},precision:{test_precision}')
