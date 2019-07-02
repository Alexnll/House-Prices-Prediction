# Kaggle: 房价预测
# 基于mxnet
# d2l.ai
import numpy as np
import pandas as pd

from mxnet import nd, gluon, init, autograd
from mxnet.gluon import data as gdata, loss as gloss, nn

path = '.\dataset\\'

# 读取训练集和测试集
def read_data(path=path):
    test_data = pd.read_csv(path + 'test.csv')
    train_data = pd.read_csv(path + 'train.csv')
    # 查看数据
    # print("test data: ")
    # print(test_data.head())
    # print(test_data.shape)
    # print()
    # print("train data: ")
    # print(train_data.head())
    # print(train_data.shape)
    
    return train_data, test_data

# 数据前处理
def data_pretreat(train, test):
    # 连接两个数据集中除saleprice的列，去除Id列（第一列），即不使用其作为训练
    all_features = pd.concat((train.iloc[: , 1:-1], test.iloc[: , 1:]))
    
    # 对数据集中的连续数值特征进行标准化
    all_features = all_features
    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
    all_features[numeric_features] = all_features[numeric_features].apply(lambda x:(x-x.mean()/x.std()))
    all_features[numeric_features] = all_features[numeric_features.fillna(0)] # 均值化后用0来替换缺失值
    # print(all_features[numeric_features])
    
    # 将离散数值转化为指示特征
    all_features = pd.get_dummies(all_features, dummy_na=True)
    # print(all_features.shape)
    return all_features

# 对mxnet，将numpy数据转化为NDArray格式，方便后续训练
def trans_type(train, test, all_features):
    n_train = train.shape[0]
    train_features = nd.array(all_features[:n_train].values)
    test_features = nd.array(all_features[n_train:].values)
    train_labels = nd.array(train.SalePrice.values).reshape((-1, 1))
    return train_features, test_features, train_labels

# 定义训练模型进行训练
# 定义网络
# 无隐藏层的感知机网络
def get_net(): 
    net = nn.Sequential()
    net.add(nn.Dense(1))   # 添加一层全连接层
    net.initialize()
    return net

# 定义损失函数
def get_loss():
    return gloss.L2Loss()

# 定义比赛中用于评价模型的对数均方根误差
def log_rmse(features, labels, net, loss):
    print('1   ', net.collect_params())
    print('2   ', features)
    print('3   ', net(features))
    clipped_preds = nd.clip(net(features), 1, float('inf'))
    rmse = nd.sqrt((2 * loss(clipped_preds.log(), labels.log()) ).mean())
    return rmse.asscalar()

# 保存训练得到的网络参数
def save_params(net):
    net.save_parameters('.\\net_parameter\\House_Price_Params')

# 导入训练得到的网络参数
def read_params():
    net_read = get_net()
    net_read.load_parameters('.\\net_parameter\\House_Price_Params')
    return net_read

# 训练
# 传入参数：训练集与测试集，超参数，网络模型和损失函数
def train(train_features, train_labels, test_features, test_labels, num_epochs, learning_rate, weight_decay, batch_size, net_in):
    train_ls, test_ls = [], []
    train_iter = gdata.DataLoader(gdata.ArrayDataset(train_features, train_labels), batch_size=batch_size, shuffle=True)

    net = net_in

    # 采用Adam优化
    loss=get_loss()
    trainer = gluon.Trainer(net.collect_params(), optimizer='adam', optimizer_params={'learning_rate': learning_rate, 'wd': weight_decay})
    
    # 开始训练
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)

        train_ls.append(log_rmse(train_features, train_labels, net, loss=loss))
        if test_labels is not None:
            test_ls.append(log_rmse(test_features, test_labels, net, loss=loss))

    # 保存训练得到的网络参数
    # save_or_not = input("Save the trained net? Y/N")
    # if save_or_not == 'Y':
    #     save_params(net)

    return train_ls, test_ls

# 验证
# 采用K折交叉验证
# 获取第i折验证时所需的训练和验证数据
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        # 按index切片
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:        
            X_train = nd.concat(X_train, X_part, dim=0)
            y_train = nd.concat(y_train, y_part, dim=0)
            
    return X_train, y_train, X_valid, y_valid
   
# 训练K次并返回训练和验证的平均误差
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        X_train, y_train, X_valid, y_valid = get_k_fold_data(k, i, X_train, y_train)
        net=get_net()
        # 主训练位置
        train_ls, valid_ls = train(X_train, y_train, X_valid, y_valid, num_epochs, learning_rate, weight_decay, batch_size, net)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]

        print('fold %d, train rmse %f, valid rmse %f' % (i, train_ls[-1], valid_ls[-1]))
    return train_l_sum / k, valid_l_sum / k

# 预测
def train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, learning_rate, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(train_features, train_labels, None, None, num_epochs, learning_rate, weight_decay, batch_size, net)
    print('train rmse: %f' % train_ls)

    preds = net(test_features).asnumpy()
    # 在原test_data后新增SalePrice列
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    prediction_data = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    prediction_data.to_csv('submission.csv', index=False)


# 主方法
def pred_main(k, num_epochs, learning_rate, weight_decay, batch_size):
    train_data, test_data = read_data()
    train_features, test_features, train_labels = trans_type(train_data, test_data, data_pretreat(train_data, test_data))
    # 获取k折交叉验证得到的平均误差
    train_l, valid_l = k_fold(k=k, X_train= train_features, y_train=train_labels, num_epochs=num_epochs, learning_rate=learning_rate, weight_decay=weight_decay, batch_size=batch_size)
    print('%d-fold validation: avg train rmse %f, avg valid rmse %f' % (k, train_l, valid_l))
    
    # 是否预测
    if_pred = input("Start predict the house prices for the test dataset? Y/N ")
    if if_pred == 'Y':
        train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, learning_rate, weight_decay, batch_size)



if __name__ == "__main__":
    # 定义超参数
    # 用于训练
    num_epochs, learning_rate, weight_decay, batch_size = 5, 0.01, 0.1, 64
    # 用于验证
    k = 10

    pred_main(k=k, num_epochs=num_epochs, learning_rate=learning_rate, weight_decay=weight_decay, batch_size=batch_size)


    未完成