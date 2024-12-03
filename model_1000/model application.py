import os
import paddle
import numpy as np
import paddle.nn as nn
import paddle.optimizer as optim
import paddle.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split

class CustomModel(nn.Layer):
    def __init__(self):
        super(CustomModel, self).__init__()

        # 添加 Dropout 层来避免过拟合
        self.dropout = nn.Dropout(p=0.12)

        # 定义第一部分：256 -> 16 将输入特征从256维降到16维。
        self.fc1 = nn.Linear(256, 16)
        self.relu1 = nn.ReLU()  # 非线性变换

        # 定义第二部分：16 -> 16
        self.fc2 = nn.Linear(16, 16)
        self.relu2 = nn.ReLU()

        # 定义第三部分：3 -> 16
        self.fc3 = nn.Linear(3, 16)
        self.relu3 = nn.ReLU()

        # 定义最终输出层：16*3 -> 2
        # 将前面三个部分处理后的特征（总共48维）转换为最终的输出，这里的输出维度是2，表示模型的输出是一个二分类问题的概率。
        self.fc_final = nn.Linear(16 * 3, 1)

    def forward(self, x1, x2, x3):
        # 处理第一个输入
        x1 = self.relu1(self.fc1(x1))

        # 处理第二个输入
        x2 = self.relu2(self.fc2(x2))

        # 处理第三个输入
        x3 = self.relu3(self.fc3(x3))

        # 拼接处理后的三个输入
        x_concat = paddle.concat([x1, x2, x3], axis=1)

        # 通过最终的全连接层
        out = self.fc_final(x_concat)

        return out

df1 = pd.read_excel('all_分子指纹.xlsx',sheet_name='指纹')
df2 = pd.read_excel('all_分子指纹.xlsx',sheet_name='电子')
df3 = pd.read_excel('all_分子指纹.xlsx',sheet_name='三特征')

x1_train = df1.iloc[:, 1:].values.astype('float32')  # 256位分子指纹
x2_train = df2.iloc[:, :].values.astype('float32')  # 16位中心金属价电子排布
x3_train = df3.iloc[:, :3].values.astype('float32') # 空速、温度、HCL

# 将DataFrame转换为Tensor
x1_train_tensor = paddle.to_tensor(x1_train)
x2_train_tensor = paddle.to_tensor(x2_train)
x3_train_tensor = paddle.to_tensor(x3_train)

print("x1_train_tensor shape:", x1_train_tensor.shape)  # 应为 [样本数, 256]
print("x2_train_tensor shape:", x2_train_tensor.shape)  # 应为 [样本数, 16]
print("x3_train_tensor shape:", x3_train_tensor.shape)  # 应为 [样本数, 3]

'''
# 加载模型参数
model = CustomModel()
model_path = "../model_500_全/model_parameters_0.pdparams"
model_state_dict = paddle.load(model_path)
model.set_state_dict(model_state_dict)

# 设置模型为评估模式
model.eval()
with paddle.no_grad():
    # 对全部数据进行预测
    outputs = model(x1_train_tensor, x2_train_tensor, x3_train_tensor)
    probabilities = F.sigmoid(outputs)
    predictions = (probabilities > 0.5).numpy().flatten()

# 将预测结果保存到Excel
df_predictions = pd.DataFrame({
    "预测概率": probabilities.numpy().flatten(),
    "预测结果": predictions
})

df_predictions.to_excel('./预测结果.xlsx', index=False)
print("预测结果已保存到预测结果.xlsx文件中")

'''

# 定义文件路径和模型数量
excel_path = './new_model_pre.xlsx'
model_dir = "../new_model_1000/"
num_models = 10

# 创建一个新文件并初始化工作表
with pd.ExcelWriter(excel_path, mode='w') as writer:
    writer.book.create_sheet('Sheet1')  # 创建初始工作表

# 逐个加载模型参数进行预测
for i in range(num_models):
    model = CustomModel()
    model_path = f"{model_dir}/model_parameters_{i}.pdparams"
    model_state_dict = paddle.load(model_path)
    model.set_state_dict(model_state_dict)

    # 设置模型为评估模式并进行预测
    model.eval()
    with paddle.no_grad():
        outputs = model(x1_train_tensor, x2_train_tensor, x3_train_tensor)
        probabilities = F.sigmoid(outputs)
        predictions = (probabilities > 0.5).numpy().flatten()

    # 保存预测结果到DataFrame
    df_predictions = pd.DataFrame({
        "模型编号": [i] * len(predictions),
        "预测概率": probabilities.numpy().flatten(),
        "预测结果": predictions
    })

    # 追加保存到已有的Excel文件中
    with pd.ExcelWriter(excel_path, mode='a', if_sheet_exists='overlay') as writer:
        # 找到当前数据写入的行数，确保数据追加到后面
        startrow = writer.sheets['Sheet1'].max_row
        df_predictions.to_excel(writer, sheet_name='Sheet1', index=False, header=False, startrow=startrow)

print("所有模型的预测结果已追加到预测结果_1000_全.xlsx文件中")
