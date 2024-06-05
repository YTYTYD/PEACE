import pandas as pd
import numpy as np
import pickle
"""
依赖经过数据处理的df_A，生成回归的pkl
"""
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 1000)

restored_df_A = pd.read_pickle('./df_A.pkl')

print(restored_df_A.shape)

grouped = restored_df_A.groupby('姓名')
all_feature_list = []
all_label_list = []
all_attention_list = []

all_regression_list_1 = []
all_regression_list_2 = []
all_regression_list_3 = []
all_regression_list_4 = []
all_regression_list_5 = []
all_regression_list_6 = []
all_regression_list_7 = []
for _, group_data in grouped:
    # group_list = group_data.drop('a', axis=1).values.tolist()

    feature_list = group_data.drop(
        [ 'Pain Relief Status','ERSO_Recom','IRSO_Recom','ERWO_Recom','IRWO_Recom','NSAIDs_Recom','A/A_Recom','Others_Recom'], axis=1).values.tolist()
    label_list = group_data['Pain Relief Status'].values.tolist()

    regression_list_1 = group_data['ERSO_Recom'].values.tolist()
    regression_list_2 = group_data['IRSO_Recom'].values.tolist()
    regression_list_3 = group_data['ERWO_Recom'].values.tolist()
    regression_list_4 = group_data['IRWO_Recom'].values.tolist()
    regression_list_5 = group_data['NSAIDs_Recom'].values.tolist()
    regression_list_6 = group_data['A/A_Recom'].values.tolist()
    regression_list_7 = group_data['Others_Recom'].values.tolist()

    attention_list = [1 for _ in range(len(label_list))]
    while len(feature_list) < 47: # 每个患者最多有47条记录，因此统一每个患者的最大token数量为47
        feature_list.append([0] * len(feature_list[0]))  # 假设所有列都填充0

    if len(label_list) < 47:
        label_list = label_list + [0] * (47 - len(label_list))

        regression_list_1 = regression_list_1 + [0] * (47 - len(regression_list_1))
        regression_list_2 = regression_list_2 + [0] * (47 - len(regression_list_2))
        regression_list_3 = regression_list_3 + [0] * (47 - len(regression_list_3))
        regression_list_4 = regression_list_4 + [0] * (47 - len(regression_list_4))
        regression_list_5 = regression_list_5 + [0] * (47 - len(regression_list_5))
        regression_list_6 = regression_list_6 + [0] * (47 - len(regression_list_6))
        regression_list_7 = regression_list_7 + [0] * (47 - len(regression_list_7))


    if len(attention_list) < 47:
        attention_list = attention_list + [0] * (47 - len(attention_list))

    all_feature_list.append(feature_list)
    all_label_list.append(label_list)
    all_attention_list.append(attention_list)

    all_regression_list_1.append(regression_list_1)
    all_regression_list_2.append(regression_list_2)
    all_regression_list_3.append(regression_list_3)
    all_regression_list_4.append(regression_list_4)
    all_regression_list_5.append(regression_list_5)
    all_regression_list_6.append(regression_list_6)
    all_regression_list_7.append(regression_list_7)

print(len(all_feature_list))
print(len(all_label_list))
print(len(all_attention_list))

print(len(all_regression_list_1))
print(len(all_regression_list_2))
print(len(all_regression_list_3))
print(len(all_regression_list_4))
print(len(all_regression_list_5))
print(len(all_regression_list_6))
print(len(all_regression_list_7))

# 打开一个文件用于二进制写入，生成分类pkl
with open('./classify.pkl', 'wb') as file:
    # 使用pickle.dump()将列表保存到文件
    pickle.dump((all_feature_list, all_label_list, all_attention_list), file)

# 打开一个文件用于二进制写入，生成回归pkl
with open('./regression_1.pkl', 'wb') as file:
    # 使用pickle.dump()将列表保存到文件
    pickle.dump((all_feature_list, all_regression_list_1, all_attention_list), file)

# 打开一个文件用于二进制写入，生成回归pkl
with open('./regression_2.pkl', 'wb') as file:
    # 使用pickle.dump()将列表保存到文件
    pickle.dump((all_feature_list, all_regression_list_2, all_attention_list), file)

# 打开一个文件用于二进制写入，生成回归pkl
with open('./regression_3.pkl', 'wb') as file:
    # 使用pickle.dump()将列表保存到文件
    pickle.dump((all_feature_list, all_regression_list_3, all_attention_list), file)

# 打开一个文件用于二进制写入，生成回归pkl
with open('./regression_4.pkl', 'wb') as file:
    # 使用pickle.dump()将列表保存到文件
    pickle.dump((all_feature_list, all_regression_list_4, all_attention_list), file)

# 打开一个文件用于二进制写入，生成回归pkl
with open('./regression_5.pkl', 'wb') as file:
    # 使用pickle.dump()将列表保存到文件
    pickle.dump((all_feature_list, all_regression_list_5, all_attention_list), file)

# 打开一个文件用于二进制写入，生成回归pkl
with open('./regression_6.pkl', 'wb') as file:
    # 使用pickle.dump()将列表保存到文件
    pickle.dump((all_feature_list, all_regression_list_6, all_attention_list), file)

# 打开一个文件用于二进制写入，生成回归pkl
with open('./regression_7.pkl', 'wb') as file:
    # 使用pickle.dump()将列表保存到文件
    pickle.dump((all_feature_list, all_regression_list_7, all_attention_list), file)

