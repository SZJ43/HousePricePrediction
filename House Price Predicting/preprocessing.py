# --coding:utf-8 --
import pandas as pd
import torch

from readData import all_features, train_data

# 若无法获得测试数据，可以通过训练数据计算均值和标准差
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean() / (x.std()))
)

# 在标准化数据之后，所有均值消失，因此可以酱缺失值设置为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# "dummies_na=True"将"na"（缺失值）视为有效的特征值，并为其创建指示符特征
all_features = pd.get_dummies(all_features, dummy_na=True)
all_features.shape

n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)