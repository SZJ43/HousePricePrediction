# --coding:utf-8 --
from matplotlib import pyplot as plt
from preprocessing import train_features, train_labels
from train import k_fold

k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0.01, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)
print(f'{k}-折验证：平均训练log rmse：{float(train_l): f},'
      f'平均验证log rmse：{float(valid_l): f}')
plt.show()