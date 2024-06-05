import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, make_scorer
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, make_scorer
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
import numpy as np

# 读取CSV文件，假设文件名为data.csv
df = pd.read_csv(r'E:\Program Files (x86)\Desktop\论文\NIPS\AAAAA提交材料\data\all_data.csv', index_col='ID')

# 删除指定的列
columns_to_drop = [ 'Medical Record Date', 'ERSO_Recom', 'IRSO_Recom', 'ERWO_Recom', 'IRWO_Recom', 'NSAIDs_Recom', 'A/A_Recom', 'Others_Recom']
df.drop(columns=columns_to_drop, inplace=True)

# 删除“疼痛缓解情况及用药后疼痛评分（4.完全缓解4.部分缓解3.轻度缓解4.无效）”列空缺的行
df.dropna(subset=['Pain Relief Status'], inplace=True)

print(df.shape)

# 将“疼痛缓解情况及用药后疼痛评分（4.完全缓解4.部分缓解3.轻度缓解4.无效）”列作为标签y，其余列作为特征x
y = df['Pain Relief Status']
x = df.drop(columns=['Pain Relief Status'])
x.fillna(0, inplace=True)


# 打印特征和标签
print("Features (x):")
print(x.shape)
print("Labels (y):")
print(y.shape)

# 检查标签的唯一值和数据类型
print(y.unique())
print(y.dtype)

# 将标签转换为整数类型（如果需要）
y = y.astype(int)


unique_indices = df.index.unique()
train_indices, test_indices = train_test_split(unique_indices, test_size=0.2, random_state=42)

# 打印索引数量
print(f'Unique indices: {len(unique_indices)}')
print(f'Train indices: {len(train_indices)}')
print(f'Test indices: {len(test_indices)}')

x_train = x.loc[train_indices]
y_train = y.loc[train_indices]
x_test = x.loc[test_indices]
y_test = y.loc[test_indices]

# 标准化特征
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 定义多个分类算法
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=10000),
    'Random Forest': RandomForestClassifier(random_state=42),
    'LinearSVC': LinearSVC(random_state=42, max_iter=10000),
    'MLP': MLPClassifier(random_state=42, max_iter=100)
}


# 定义评估函数
def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')

    # 对于多分类问题，计算ROC AUC需要将标签二值化
    lb = LabelBinarizer()
    y_test_binarized = lb.fit_transform(y_test)
    y_pred_binarized = lb.transform(y_pred)
    auroc = roc_auc_score(y_test_binarized, y_pred_binarized, average='macro', multi_class='ovo')

    return acc, f1, recall, precision, auroc


# 定义评估指标
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'f1_macro': make_scorer(f1_score, average='macro'),
    'recall_macro': make_scorer(recall_score, average='macro'),
    'precision_macro': make_scorer(precision_score, average='macro'),
    'roc_auc_ovo': make_scorer(roc_auc_score, average='macro', multi_class='ovo', needs_proba=True)
}

# 训练和评估每个模型，并打印结果
for name, model in models.items():
    print(f'\n{name} Cross-validation Results:')
    # 5折交叉验证
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold = 1
    for train_index, val_index in cv.split(x_train, y_train):
        x_fold_train, x_fold_val = x_train[train_index], x_train[val_index]
        y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]

        print(f'Fold {fold} - Train indices: {len(train_index)}, Val indices: {len(val_index)}')

        model.fit(x_fold_train, y_fold_train)
        acc, f1, recall, precision, auroc = evaluate_model(model, x_fold_val, y_fold_val)
        print(
            f'Fold {fold} - Accuracy: {acc:.4f}, F1 Score: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, AUROC: {auroc:.4f}')
        fold += 1

    # 在训练集上训练模型
    model.fit(x_train, y_train)

    # 在测试集上评估模型
    acc, f1, recall, precision, auroc = evaluate_model(model, x_test, y_test)
    print(f'\n{name} Test Results:')
    print(f'Accuracy: {acc:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'AUROC: {auroc:.4f}')



y = y - 1


# 定义多个分类算法
models = {
    'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
    'LightGBM': lgb.LGBMClassifier(random_state=42),
    'AdaBoost': AdaBoostClassifier(random_state=42)

}


# 定义评估函数
def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')

    # 对于多分类问题，计算ROC AUC需要将标签二值化
    lb = LabelBinarizer()
    y_test_binarized = lb.fit_transform(y_test)
    y_pred_binarized = lb.transform(y_pred)
    auroc = roc_auc_score(y_test_binarized, y_pred_binarized, average='macro', multi_class='ovo')

    return acc, f1, recall, precision, auroc


# 定义评估指标
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'f1_macro': make_scorer(f1_score, average='macro'),
    'recall_macro': make_scorer(recall_score, average='macro'),
    'precision_macro': make_scorer(precision_score, average='macro'),
    'roc_auc_ovo': make_scorer(roc_auc_score, average='macro', multi_class='ovo', needs_proba=True)
}

# 训练和评估每个模型，并打印结果
for name, model in models.items():
    print(f'\n{name} Cross-validation Results:')
    # 5折交叉验证
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold = 1
    for train_index, val_index in cv.split(x_train, y_train):
        x_fold_train, x_fold_val = x_train[train_index], x_train[val_index]
        y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]
        model.fit(x_fold_train, y_fold_train)
        acc, f1, recall, precision, auroc = evaluate_model(model, x_fold_val, y_fold_val)
        print(
            f'Fold {fold} - Accuracy: {acc:.4f}, F1 Score: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, AUROC: {auroc:.4f}')
        fold += 1

    # 在训练集上训练模型
    model.fit(x_train, y_train)

    # 在测试集上评估模型
    acc, f1, recall, precision, auroc = evaluate_model(model, x_test, y_test)
    print(f'\n{name} Test Results:')
    print(f'Accuracy: {acc:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'AUROC: {auroc:.4f}')