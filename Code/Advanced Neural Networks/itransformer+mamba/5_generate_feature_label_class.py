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

restored_df_A = pd.read_pickle('./df_class.pkl')

print(restored_df_A.shape)

grouped = restored_df_A.groupby('ID')
all_feature_list = []
all_label_list = []
all_attention_list = []

max_token_length = 12 # 去掉缺失记录后，每个患者最多有12条记录，因此统一每个患者的最大token数量为12
for _, group_data in grouped:
    # group_list = group_data.drop('a', axis=1).values.tolist()

    feature_list = group_data.drop(
        ['ID', 'Medical Record Date', 'Pain Relief Status','ERSO_Recom','IRSO_Recom','ERWO_Recom','IRWO_Recom','NSAIDs_Recom','A/A_Recom','Others_Recom'], axis=1).values.tolist()
    label_list = group_data['Pain Relief Status'].values.tolist()
    print(label_list)
    print(len(feature_list[0]))

    attention_list = [1 for _ in range(len(label_list))]
    while len(feature_list) < max_token_length: # 去掉缺失记录后，每个患者最多有12条记录，因此统一每个患者的最大token数量为12
        feature_list.append([0] * len(feature_list[0]))  # 假设所有列都填充0

    if len(label_list) < max_token_length:
        label_list = label_list + [0] * (max_token_length - len(label_list))


    if len(attention_list) < max_token_length:
        attention_list = attention_list + [0] * (max_token_length - len(attention_list))

    assert len(feature_list) == max_token_length, "feature_list长度不是12"
    assert len(label_list) == max_token_length, "label_list长度不是12"
    assert len(attention_list) == max_token_length, "attention_list长度不是12"


    all_feature_list.append(feature_list)
    all_label_list.append(label_list)
    all_attention_list.append(attention_list)


print(len(all_feature_list))
print(len(all_label_list))
print(len(all_attention_list))


# 打开一个文件用于二进制写入，生成分类pkl
with open('./classify_2.pkl', 'wb') as file:
    # 使用pickle.dump()将列表保存到文件
    pickle.dump((all_feature_list, all_label_list, all_attention_list), file)

