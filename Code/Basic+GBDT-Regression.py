import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import LinearSVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error
import lightgbm as lgb
import xgboost as xgb

# 读取CSV文件并将“ID”列设置为索引
df = pd.read_csv(r'E:\Program Files (x86)\Desktop\论文\NIPS\AAAAA提交材料\data\all_data.csv', index_col='ID')

# 删除不需要的列
columns_to_drop = ['Pain Relief Status', 'Medical Record Date']
df = df.drop(columns=columns_to_drop)

# 将指定列作为标签
labels = [ 'ERSO_Recom', 'IRSO_Recom', 'ERWO_Recom', 'IRWO_Recom', 'NSAIDs_Recom', 'A/A_Recom', 'Others_Recom']
y = df[labels]

# 其余列作为特征
x = df.drop(columns=labels)

# 处理空字符串和非数值字符，将其替换为NaN，然后再填充为0（或者其他适合的值）
x = x.replace(' ', np.nan).astype(float)
x = x.fillna(0)


# 将标签转换为整数类型（如果需要）
y = y.astype(int)

# 打印特征和标签以验证结果
print("Features (x):")
print(x.head())

print("\nLabels (y):")
print(y.head())

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 按索引进行8:2划分训练集和测试集
unique_indices = df.index.unique()
train_indices, test_indices = train_test_split(unique_indices, test_size=0.2, random_state=42)

x_train = x.loc[train_indices]
x_test = x.loc[test_indices]
y_train = y.loc[train_indices]
y_test = y.loc[test_indices]

# 打印索引数量
print(f'Unique indices: {len(unique_indices)}')
print(f'Train indices: {len(train_indices)}')
print(f'Test indices: {len(test_indices)}')


# 标准化特征
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 打印结果以验证
print(" (x_train):",x_train.shape)
print(" (x_test):",x_test.shape)
print(" (y_train):",y_train.shape)
print(" (y_test):",y_test.shape)



# 自定义评估函数
scoring = {'mse': make_scorer(mean_squared_error), 'mae': make_scorer(mean_absolute_error)}

# 定义模型
models = {
    'DecisionTreeRegressor': DecisionTreeRegressor(random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=10000),
    'MLP': MLPRegressor(random_state=42, max_iter=10000),
    'LightGBM': lgb.LGBMRegressor(random_state=42),
    'XGBoost': xgb.XGBRegressor(random_state=42),
    'AdaBoost': AdaBoostRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'LinearSVR': LinearSVR(random_state=42)
}

# 对每种标签和每种模型进行训练和评估
for label in labels:
    print(f"Evaluating label: {label}\n")
    for model_name, model in models.items():
        print(f"Model: {model_name}")

        # 进行5折交叉验证
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        fold_idx = 1
        for train_index, val_index in kf.split(x_train):
            print(f"Fold {fold_idx}  Train indices: {len(train_index)}, Val indices: {len(val_index)}")
            fold_idx += 1

        cv_results = cross_validate(model, x_train, y_train[label], cv=kf, scoring=scoring, return_train_score=True,
                                    n_jobs=-1)

        # 打印每一折的MSE和MAE
        for fold_idx in range(5):
            print(
                f"Fold {fold_idx + 1}  Val MSE: {cv_results['test_mse'][fold_idx]:.4f}, Val MAE: {cv_results['test_mae'][fold_idx]:.4f}")

        # 训练模型
        model.fit(x_train, y_train[label])

        # 预测测试集
        y_pred_test = model.predict(x_test)

        # 计算测试集上的MSE和MAE
        mse_test = mean_squared_error(y_test[label], y_pred_test)
        mae_test = mean_absolute_error(y_test[label], y_pred_test)

        # 输出测试集结果
        print(f"  Test Set MSE: {mse_test:.4f}, Test Set MAE: {mae_test:.4f}\n")
    print("-" * 50)