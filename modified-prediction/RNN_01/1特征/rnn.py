import random
import shutil

import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# TF2描述循环计算层

# 读取贵州茅台日k线数据到变量maotai
maotai = pd.read_csv('../../stock_data.csv')  # 读取股票文件
wuliangye = pd.read_csv('../../五粮液stock_data.csv')
beidahuang = pd.read_csv('../../北大荒stock_data.csv')
huzhoulaojiao = pd.read_csv('../../泸州老窖stock_data.csv')
yutongkeji = pd.read_csv('../../裕同科技stock_data.csv')

training_set = pd.concat([maotai.iloc[0:1529 - 300, 3:4],
                          # wuliangye.iloc[0:1529 - 300, 3:4],
                          # beidahuang.iloc[0:1529 - 300, 3:4],
                          # huzhoulaojiao.iloc[0:1529 - 300, 3:4],
                          # yutongkeji.iloc[0:1529 - 300, 3:4]
                          ], axis=1).values
test_set = pd.concat([maotai.iloc[1529 - 300:, 3:4],
                      # wuliangye.iloc[1529 - 300:, 3:4],
                      # beidahuang.iloc[1529 - 300:, 3:4],
                      # huzhoulaojiao.iloc[1529 - 300:, 3:4],
                      # yutongkeji.iloc[1529 - 300:, 3:4]
                      ], axis=1).values # 后300天的开盘价作为测试集


sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
test_set_scaled = sc.transform(test_set)


x_train = []
y_train = []
x_test = []
y_test = []


for i in range(60, len(training_set_scaled)):
    x_train.append(training_set_scaled[i - 60:i, :])
    y_train.append(training_set_scaled[i, 0])


# 合并 x_train 和 y_train
combined_data = list(zip(x_train, y_train))

# 随机打乱数据
random.shuffle(combined_data)

# 将打乱后的数据重新分开为 x_train 和 y_train
x_train_shuffled, y_train_shuffled = zip(*combined_data)

# 转换为列表
x_train = list(x_train_shuffled)
y_train = list(y_train_shuffled)

# 将训练集由list格式变为array格式
x_train, y_train = np.array(x_train), np.array(y_train)


x_train = np.reshape(x_train, (x_train.shape[0], 60, 1))


for i in range(60, len(test_set_scaled)):
    x_test.append(test_set_scaled[i - 60:i, :])
    y_test.append(test_set_scaled[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], 60, 1))





####################################################

import tensorflow as tf
# from keras.layers import , 
from tensorflow.keras.layers import SimpleRNN,Dropout,Dense
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

# 用Sequential搭建神经网络
model = tf.keras.Sequential([
    SimpleRNN(80, return_sequences=True),  # 第一层循环计算层，记忆体设定80个；每个时间步推送ht给下一层
    Dropout(0.2),
    SimpleRNN(100),  # 第二层循环计算层，记忆体设定100个；仅最后的时间步推送ht给下一层
    Dropout(0.2),
    Dense(1)   # 输出值是第61天的开盘价，只有一个数
])

# 配置训练方法
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),  # 学习率
              loss='mean_squared_error')  # 损失函数用均方误差


checkpoint_save_path = "./checkpoint_rnn/"

# 检查目录是否存在
if os.path.exists(checkpoint_save_path):
    print("目录已存在，正在刷新...")
    # 这里可以执行刷新操作，比如清空目录下的所有文件
    for file_name in os.listdir(checkpoint_save_path):
        file_path = os.path.join(checkpoint_save_path, file_name)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)  # 删除文件
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # 删除子目录及其内容
        except Exception as e:
            print(f"无法删除 {file_path}: {e}")
else:
    # 创建目录
    os.makedirs(checkpoint_save_path)
    print("目录不存在，已创建。")

if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 monitor='val_loss')  # 由于这里不观测acc，不计算测试集准确率，故根据val_loss保存最优模型

# fit执行训练过程
history = model.fit(x_train, y_train, batch_size=64, epochs=200, validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=[cp_callback]) 

model.summary()   # 打印出网络结构和参数统计

# 参数提取
file = open('./rnn_stock_weights.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

loss = history.history['loss']
val_loss = history.history['val_loss']

# loss可视化
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('rnn_loss_compare.png')  # 保存图像为 PNG 格式
plt.clf()
################## predict ######################
# 测试集输入模型进行预测
predicted_stock_price = model.predict(x_test)
# print(predicted_stock_price)

# 对预测数据还原---从（0，1）反归一化到原始范围
min,scale = sc.min_,sc.scale_
predicted_stock_price = (predicted_stock_price-min[0])/scale[0]


# real_stock_price = sc.inverse_transform(test_set[60:])   # 240组测试数据
real_stock_price = test_set[60:,0]

# 画出真实数据和预测数据的对比曲线
plt.plot(real_stock_price, color='red', label='MaoTai Stock Price')   # 真实值曲线
plt.plot(predicted_stock_price, color='blue', label='Predicted MaoTai Stock Price')  # 预测值曲线
plt.title('MaoTai Stock Price Prediction')
plt.xlabel('Time(day)')
plt.ylabel('MaoTai Stock Price(yuan)')
plt.legend()
plt.savefig('rnn_stock_prediction_plot.png')  # 保存图像为 PNG 格式


########## evaluate ##############
'''
为了评价模型优劣，给出三个评判指标：均方误差MSE  均方根误差RMSE  平均绝对误差MAE
这些误差越小，说明预测的数值与真实值越接近
'''
# calculate MSE 均方误差 ---> E[(预测值-真实值)^2] (预测值减真实值求平方后求均值)
mse = mean_squared_error(predicted_stock_price, real_stock_price)
# calculate RMSE 均方根误差--->sqrt[MSE]    (对均方误差开方)
rmse = math.sqrt(mean_squared_error(predicted_stock_price, real_stock_price))
# calculate MAE 平均绝对误差----->E[|预测值-真实值|](预测值减真实值求绝对值后求均值）
mae = mean_absolute_error(predicted_stock_price, real_stock_price)

print('均方误差: %.6f' % mse)
print('均方根误差: %.6f' % rmse)
print('平均绝对误差: %.6f' % mae)