# -*- coding: utf-8 -*-
"""
Created on 2023-01
@author: ZQ
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设置使用的GPU设备（例如这里选择GPU 0）
import time
import random
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from make_dataset import *  # 导入自定义的数据集处理模块
from make_model import *  # 导入自定义的模型创建模块
from Utils import *  # 导入辅助函数模块
import tensorflow as tf

# 打印开始信息
print("[  Respiratory Rate Prediction Starts  ]")

# 配置参数
LOAD_FORM_SAVE = False  # 是否从保存文件加载模型
LOAD_MODEL = False  # 是否加载已有模型
WIN_SIZE = 125*16  # 窗口大小
BATCH_SIZE = 64  # 批大小
EPOCHS = 500  # 训练轮数
DOWN_SAMPLING_GRADE = 8  # 数据下采样级别
LR = 0.001  # 学习率
CNN_FILTERS = 512  # CNN层的滤波器数量
CNN_KERNEL = 20  # CNN卷积核大小
LSTM_UNIT = 256  # LSTM单元数
LSTM_DENSE = 1024  # LSTM层的全连接单元数
DENSE1_DIM = 512  # 第一层全连接层的维度
DENSE2_DIM = 128  # 第二层全连接层的维度
MAX = 1000000000  # 随机种子的最大值
FOLD_NUM = 10  # 交叉验证的折数

# 训练过程中使用的回调函数
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=5, mode='auto', min_lr=0.0001)
early_stop = EarlyStopping(monitor="val_loss", patience=20, verbose=0, mode="min")  # 早停策略

# 数据集路径（CSV文件）
csv_path = '/home/zz/respiratory_rate_prediction/data/bidmc_RR_16s_overlap87.5_vmd_zscore_RRscreen.csv'
# csv_path = '/home/zz/respiratory_rate_prediction/data/capnobase_RR_16s_overlap87.5_vmd_zscore_RRscreen_age5.csv'

# 读取数据
raw_data = read_csv(WIN_SIZE, csv_path)

# 进行K折交叉验证
for fold_index in range(FOLD_NUM):
    # 切分数据为训练、验证、测试集
    input_train_np, input_val_np, input_test_np = fold_n(fold_index=fold_index, raw_data=raw_data, fold_num=FOLD_NUM)

    # 构建数据集
    x1_train, x2_train, y_train, x1_val, x2_val, y_val, x1_test, x2_test, y_test = make_dataset_from_fold_n(
        win_size=WIN_SIZE, input_train_np=input_train_np, input_val_np=input_val_np, input_test_np=input_test_np)

    # 对数据进行下采样
    x1_train, x2_train, y_train, x1_val, x2_val, y_val, x1_test, x2_test, y_test = down_sampling(
        x1_train, x2_train, y_train, x1_val, x2_val, y_val, x1_test, x2_test, y_test, down_sampling_grade=DOWN_SAMPLING_GRADE)

    # 进行多次重复训练
    for repeat_index in range(1):
        TIME_STAMP = time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))  # 获取当前时间戳

        # 设置随机种子，确保结果可复现
        seed_value = np.random.randint(MAX)
        random.seed(seed_value)
        np.random.seed(seed_value)
        tf.random.set_seed(seed_value)
        print("The seed value in {}th training is {}".format(repeat_index, seed_value))

        # 创建模型
        model = TransRR(250)  # 这里假设TransRR是你自己定义的模型
        model.compile(optimizer=optimizers.Adam(lr=LR), loss="mae")  # 使用Adam优化器和MAE损失函数
        model.summary()  # 打印模型概述

        # 训练模型
        history = model.fit(x=[x1_train, x2_train], y=[y_train],
                            validation_data=([x1_val, x2_val], [y_val]),
                            batch_size=64, shuffle=True, epochs=EPOCHS, verbose=1,
                            callbacks=[reduce_lr, early_stop])  # 训练并使用回调函数

        # 训练历史记录
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        # 预测结果
        predicted_rr_train = model.predict(x=[x1_train, x2_train], verbose=1)
        predicted_rr_val = model.predict(x=[x1_val, x2_val], verbose=1)
        predicted_rr_test = model.predict(x=[x1_test, x2_test], verbose=1)

        # 实际值
        real_rr_train = y_train
        real_rr_val = y_val
        real_rr_test = y_test

        # 去掉多余的维度
        predicted_rr_train = predicted_rr_train.squeeze()
        real_rr_train = real_rr_train.squeeze()
        predicted_rr_val = predicted_rr_val.squeeze()
        real_rr_val = real_rr_val.squeeze()
        predicted_rr_test = predicted_rr_test.squeeze()
        real_rr_test = real_rr_test.squeeze()

        # 计算评估指标
        rr_in_train = np.array(real_rr_train)
        rr_in_val = np.array(real_rr_val)
        rr_in_test = np.array(real_rr_test)

        # 打印当前折次和重复训练的结果
        print("[ fold_index-"+str(fold_index)+"(repeat" +str(repeat_index)+") Respiratory Rate Prediction Ends ]")
        print("val_loss:" + str(round(val_loss[-1],2)))  # 打印验证集的最终损失
        print("test mae:", loss_mae(rr_in_test, predicted_rr_test))  # 打印测试集MAE
        print("test e:", loss_e(rr_in_test, predicted_rr_test))  # 打印测试集的E指标（可能是误差）
        print("test pcc:", loss_pcc(rr_in_test, predicted_rr_test))  # 打印测试集的PCC（皮尔逊相关系数）
        print("test loa:", loss_loa(rr_in_test, predicted_rr_test))  # 打印测试集LOA（差异分析）

# 打印所有训练完成的信息
print("[ ALL Respiratory Rate Prediction Ends ]")

exit(0)  # 程序结束

from Utils import *
import tensorflow as tf

print("[  Respiratory Rate Prediction Starts  ]")
LOAD_FORM_SAVE = False
LOAD_MODEL = False
WIN_SIZE = 125*16
BATCH_SIZE = 64
EPOCHS = 500
DOWN_SAMPLING_GRADE = 8
LR = 0.001
CNN_FILTERS = 512
CNN_KERNEL = 20
LSTM_UNIT = 256
LSTM_DENSE = 1024
DENSE1_DIM = 512
DENSE2_DIM = 128
MAX = 1000000000
FOLD_NUM = 10
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=5, mode='auto', min_lr=0.0001)
early_stop = EarlyStopping(monitor="val_loss", patience=20, verbose=0, mode="min")

# dataset
csv_path = '/home/zz/respiratory_rate_prediction/data/bidmc_RR_16s_overlap87.5_vmd_zscore_RRscreen.csv'
# csv_path = '/home/zz/respiratory_rate_prediction/data/capnobase_RR_16s_overlap87.5_vmd_zscore_RRscreen_age5.csv'

raw_data = read_csv(WIN_SIZE, csv_path)

for fold_index in range(FOLD_NUM):
    input_train_np, input_val_np, input_test_np = fold_n(fold_index=fold_index, raw_data=raw_data, fold_num=FOLD_NUM)

    x1_train, x2_train, y_train, x1_val, x2_val, y_val, x1_test, x2_test, y_test = make_dataset_from_fold_n(
        win_size=WIN_SIZE, input_train_np=input_train_np, input_val_np=input_val_np, input_test_np=input_test_np)

    x1_train, x2_train, y_train, x1_val, x2_val, y_val, x1_test, x2_test, y_test = down_sampling(
        x1_train, x2_train, y_train, x1_val, x2_val, y_val, x1_test, x2_test, y_test, down_sampling_grade=DOWN_SAMPLING_GRADE)

    for repeat_index in range(1):
        TIME_STAMP = time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))

        seed_value = np.random.randint(MAX)
        random.seed(seed_value)
        np.random.seed(seed_value)
        tf.random.set_seed(seed_value)
        print("The seed value in {}th training is {}".format(repeat_index, seed_value))

        model = TransRR(250)
        model.compile(optimizer=optimizers.Adam(lr=LR), loss="mae")
        model.summary()

        history = model.fit(x=[x1_train, x2_train], y=[y_train],
                            validation_data=([x1_val, x2_val], [y_val]),
                            batch_size=64, shuffle=True, epochs=EPOCHS, verbose=1,
                            callbacks=[reduce_lr, early_stop])

        loss = history.history['loss']
        val_loss = history.history['val_loss']
        predicted_rr_train = model.predict(x=[x1_train, x2_train], verbose=1)
        predicted_rr_val = model.predict(x=[x1_val, x2_val], verbose=1)
        predicted_rr_test = model.predict(x=[x1_test, x2_test], verbose=1)
        real_rr_train = y_train
        real_rr_val = y_val
        real_rr_test = y_test

        predicted_rr_train = predicted_rr_train.squeeze()
        real_rr_train = real_rr_train.squeeze()
        predicted_rr_val = predicted_rr_val.squeeze()
        real_rr_val = real_rr_val.squeeze()
        predicted_rr_test = predicted_rr_test.squeeze()
        real_rr_test = real_rr_test.squeeze()

        rr_in_train = np.array(real_rr_train)
        rr_in_val = np.array(real_rr_val)
        rr_in_test = np.array(real_rr_test)
        print("[ fold_index-"+str(fold_index)+"(repeat" +str(repeat_index)+") Respiratory Rate Prediction Ends ]")
        print("val_loss:" + str(round(val_loss[-1],2)))
        print("test mae:", loss_mae(rr_in_test, predicted_rr_test))
        print("test e:", loss_e(rr_in_test, predicted_rr_test))
        print("test pcc:", loss_pcc(rr_in_test, predicted_rr_test))
        print("test loa:", loss_loa(rr_in_test, predicted_rr_test))

print("[ ALL Respiratory Rate Prediction Ends ]")
exit(0)
