# QCAFE

## startup 程序运行入口
支持修改参数：
1. 数据集路径
2. batchsize
3. 学习率lr
4. 训练epoch数

## run 运行调用文件
### run 运行调用统一入口
### init 导入本文件夹内所有定制函数
### run_cafe 运行调用模型
支持修改参数：
1. 数据集文件名
2. 优化器
3. 训练入口 trainer.fit
4. 测试入口


## train 训练步骤文件
### trainer 训练通用步骤
### cafe_trainer 模型训练具体步骤
train_epoch结果打印

## model 模型文件
### model 抽象模型
### cafe

## tb-logs
使用命令：tensorboard --logdir=tb_logs/
查看训练数据图像