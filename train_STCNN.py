import os
import torch
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import models as models
import pickle
import time
from data_process import *

data_physical_name = './data/Tlines_S-parameter.csv'
save_img_path = './img'
sample_number = 950
test_number = 1000 - sample_number
f_interval = 500
test_schedule = 5
training_iter = 1000
batch_size = 950
cwd = os.getcwd()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
# 载入数据， 将频率参数作为输入参数
data_all = load_physical_para_data(data_physical_name)  # 读取所有矩阵参数

physical_parameter = data_all[:, :3]

physical_parameter = physical_parameter[::500, :].astype(np.float64)  # 取出其中一组物理参数
output_parameter = data_all[:, -1].reshape(-1, 500).astype(np.float64)  # 得到所有情况下所有频率的输出

# 划分训练集和测试集，训练集选择720个
train_x, train_y, test_x, test_y = divede_dataset(physical_parameter, output_parameter, sample_number)
# 将训练集合测试集的数据标准化
tensor_x, tensor_y, tensor_test_x, tensor_test_y, tensor_meanY, tensor_stdY, tensor_meanx, tensor_stdx = \
    data_pre_process(train_x, train_y, test_x, test_y)

NMSE_Test = []
modelWeights = []
NMSE_Test_Median = []

# 选择了ZTCNN型号
model = models.ZTCNN().to(device)
# 选择要使用的优化器，定义初始“学习率（lr）”，并在里程碑处定义学习率降低率（gamma）
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300, 500, 1000, 2500, 3500], gamma=0.5)
numParams = sum([p.numel() for p in model.parameters()])


def calc_training_loss(recon_y, y):
    err = (torch.abs(y - recon_y) ** 2)
    err = err.sum(dim=1) / err.shape[1]
    # loss = torch.sum((y - recon_y)**2)
    return err.sqrt().mean()


def calc_test_NMSE(recon_y, y):
    err = (torch.abs(y - recon_y) ** 2)
    err = err.sum(dim=1) / err.shape[1]
    return err.sqrt().mean()


# Closure to be called by optimizer during training.优化器
def closure(data_x, data_y):
    optimizer.zero_grad()
    output = model(data_x)

    loss = calc_training_loss(output, data_y)
    return loss


def batchdivide(size, data_x, data_y, i):
    input_batch = data_x[i * size:(i + 1) * size, :]  # 取出物理特征
    output_batch = data_y[i * size:(i + 1) * size, :]  # 取出S参数
    return input_batch, output_batch



current_time = time.time()
print(f"Starting training the model. \n")
print(f"""-----------------------------------------------------------------""")

for a in range(training_iter):
    for j in range(0, train_x.shape[0] // batch_size):
        train_data_x, train_data_y = batchdivide(batch_size, tensor_x, tensor_y, j)
        model.train()

        loss = closure(train_data_x.to(device), train_data_y.to(device))

        loss.backward()

        optimizer.step()
        scheduler.step()
        if (a % 1 == 0) and (j % 2 == 0):
            print(
                'Train Epoch : {} [{:0>4d}/{}]\tLoss: {:.6f}'.format(a, batch_size * j, len(train_data_x), loss.item()))

    if a % test_schedule == 0:
        model.eval()
        with torch.no_grad():
            test_y = tensor_test_y.to(device)
            test_x = tensor_test_x.to(device)
            test_output = model(test_x)
            NMSE = calc_test_NMSE(test_output, test_y)
            avNMSE = NMSE.mean()
            medNMSE = NMSE.median()

            modelWeights += [model.state_dict()]

            NMSE_Test += [avNMSE.item()]
            NMSE_Test_Median += [medNMSE.item()]

elapsed = time.time() - current_time
print(f"""\n-----------------------------------------------------------------""")
print(f"""训练所需时间为 {elapsed / 60 :.3f} minutes""")
start = time.time()
axis_x = np.linspace(0.1, 5e10, f_interval)
S_test_renorm, test_output_renorm = data_renorm(
    tensor_test_y.to('cpu'), test_output.to('cpu'), tensor_meanY.to('cpu'),
    tensor_stdY.to('cpu'))
end = time.time()
print("测试时间为:", end - start)
for n in range(test_number):
    acc_test = torch.abs((test_output_renorm[n, :] - S_test_renorm[n, :]) / S_test_renorm[
                                                                            n, :])

    print("测试集样本" + str(n + 1) + "的准确率为：",
          (1 - acc_test.cpu().detach().numpy().reshape(-1, ).mean()) * 100)
    plt.figure()
    plt.plot(axis_x, S_test_renorm.cpu()[n, :], 'k-', label='actual')
    plt.plot(axis_x, test_output_renorm.cpu().detach().numpy()[n, :],
             'r-', label='prediction')
    plt.xlabel('f/GHz')
    plt.ylabel('S_Parameter')
    plt.title('No' + str(n + 1))
    plt.legend()

    plt.text(10, -50, "Accuracy:" + str(
        "%.2f" % ((1 - acc_test.cpu().detach().numpy().reshape(-1, ).mean()) * 100)) + "%")

    plt.ylim(-70, 1)

    # plt.savefig(save_img_path + 'No.' + str(n + 1) + '.jpg')  # 保存图片

NMSE_Test = np.asarray(NMSE_Test)
best_idx = np.argmin(NMSE_Test)
final_model_weights = modelWeights[best_idx]

# 最终模型的权重将以名称“save_name”保存。
save_name = "STCNN_Solenoid.pth"
print(
    f"测试集上的最终模型性能 -> Average NMSE: {100 * NMSE_Test[best_idx]:.2f}%, Median NMSE: {100 * NMSE_Test_Median[best_idx]:.2f}%.")
torch.save(final_model_weights, cwd + "/" + save_name)
print(f"模型权重被保存在 \"{cwd}\\{save_name}\"")
print(f"""-----------------------------------------------------------------""")
print("Exiting.")
