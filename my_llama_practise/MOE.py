import torch  # 导入PyTorch库
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入函数模块

# 定义一个专家类，它是一个简单的全连接层
class Expert(nn.Module):
    def __init__(self, input_size, output_size):
        super(Expert, self).__init__()  # 调用父类的初始化函数
        self.fc = nn.Linear(input_size, output_size)  # 定义一个线性层，输入维度为input_size，输出维度为output_size

    def forward(self, x):
        return F.relu(self.fc(x))  # 通过线性层后应用ReLU激活函数，并返回结果

# 定义一个门控类，用于计算每个专家的门控分数
class Gate(nn.Module):
    def __init__(self, input_size, num_experts):
        super(Gate, self).__init__()  # 调用父类的初始化函数
        self.fc = nn.Linear(input_size, num_experts)  # 定义一个线性层，输入维度为input_size，输出维度为num_experts

    def forward(self, x):
        return self.fc(x)  # 直接返回线性层的输出，不使用softmax

# 定义硬性门控混合专家（Hard Mixture of Experts, MoE）模型
class HardMoE(nn.Module):
    def __init__(self, input_size, output_size, num_experts, top_k=1):
        super(HardMoE, self).__init__()  # 调用父类的初始化函数
        self.experts = nn.ModuleList([Expert(input_size, output_size) for _ in range(num_experts)])  # 创建多个专家
        self.gate = Gate(input_size, num_experts)  # 创建门控
        self.top_k = top_k  # 定义top-k参数

    def forward(self, x):
        # x的形状为 [N, S, V]，其中N是批量大小，S是序列长度，V是特征维数
        N, S, V = x.size()

        gate_output = self.gate(x)  # 计算门控输出，形状为 [N, S, num_experts]

        # 使用top_k选择专家
        top_values, top_indices = torch.topk(gate_output, self.top_k, dim=-1)
        
        # 创建一个填充的tensor，用于存储每个时间步的专家输出
        expert_outputs = torch.zeros(N, S, self.top_k, output_size, device=x.device)  # output_size 应该与专家输出一致

        # 对每个时间步选择专家
        for t in range(S):
            for i in range(N):
                for j in range(self.top_k):
                    expert_index = top_indices[i, t, j]
                    expert_outputs[i, t, j] = self.experts[expert_index](x[i, t])  # 调用选定的专家进行计算

        output = expert_outputs.mean(dim=2)  # 对top-k专家的输出取均值，形状为 [N, S, output_size]
        return output

# 示例代码使用
input_size = 10  # 输入特征大小
output_size = 5  # 输出特征大小
num_experts = 4  # 专家数量
top_k = 2  # 激活的专家数量

model = HardMoE(input_size, output_size, num_experts, top_k)  # 创建模型实例

# 随机输入数据
N = 32  # 批量大小
S = 15  # 序列长度
V = input_size  # 特征维数

x = torch.randn(N, S, V)  # 生成输入数据

# 前向传播
output = model(x)  # 通过模型进行前向传播
print(output.shape)  # 打印输出的形状，应该是 [32, 15, output_size]，即 [N, S, output_size]