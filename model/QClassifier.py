import pennylane as qml
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    # 编码输入
    # for i in range(n_qubits):
    #     qml.RY(inputs[i], wires=i)
    qml.AngleEmbedding(inputs, wires=[0, 1, 2, 3], rotation="Y")
    # qml.AmplitudeEmbedding(inputs, wires=[0, 1, 2, 3],normalize=True)
    
    # 参数化旋转层
    for i in range(n_qubits):
        qml.Rot(*weights[i], wires=i)
    
    # 全连接门
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i+1])
    
    # 输出测量
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


class QuantumClassifier(nn.Module):
    def __init__(self, n_features=96, n_qubits=4):
        super().__init__()
        self.pre_net = nn.Linear(n_features, n_qubits)  # 输入特征到量子比特的映射
        self.q_params = nn.Parameter(torch.randn(n_qubits, 3)* 0.1)  # 3参数旋转门
        self.post_net = nn.Linear(4, 2) # 输出 fake / real

    def forward(self, x):
        x = self.pre_net(x)
        x = torch.tanh(x) * np.pi  # 将输入缩放到 [-π, π] 
        q_out = torch.stack([torch.tensor(quantum_circuit(xi, self.q_params), dtype=torch.float32) for xi in x]).to(device)
        return self.post_net(q_out)
