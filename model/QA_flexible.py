import pennylane as qml
import torch
import torch.nn as nn
import numpy as np
# import matplotlib.pyplot as plt

qubits = 4  # Number of qubits for the quantum attention core mechanism
# 定义 score 和 value 用的 device
dev_score = qml.device("default.qubit", wires=qubits*2, shots=None)
dev_value = qml.device("default.qubit", wires=qubits, shots=None)

def initQKV_on_wires(rot_params, crx_params, *, offset=0): # 星号 * 表示之后的参数必须用 key=value 形式调用
    # 每个 qubit 的 Rot 需要 3 个参数
    for i in range(qubits):
        idx = i * 3
        wire = offset + i
        qml.RZ(rot_params[idx], wires=wire)
        qml.RX(rot_params[idx + 1], wires=wire)
        qml.RZ(rot_params[idx + 2], wires=wire)

    # CRX ring
    for i in range(qubits):
        control = offset + i
        target = offset + (i + 1) % qubits
        qml.CRX(crx_params[i], wires=[control, target])

    # CNOT ring
    for i in range(qubits):
        control = offset + i
        target = offset + (i + 1) % qubits
        qml.CNOT(wires=[control, target])

# qmha_score
@qml.qnode(dev_score, interface="torch", diff_method="backprop")
def qmha_score(xq, xk, weights_q_rot, weights_q_crx, weights_k_rot, weights_k_crx, weights_cross_rot, weights_cross_crx):
    qml.AngleEmbedding(xq, wires=list(range(qubits)))
    initQKV_on_wires(weights_q_rot, weights_q_crx )

    qml.AngleEmbedding(xk, wires=list(range(qubits, 2 * qubits)))
    initQKV_on_wires(weights_k_rot, weights_k_crx, offset=qubits)

    # Cross CRX 双向交叉
    for i in range(qubits):
        qml.CRX(weights_cross_crx[i], wires=[i, i + qubits])
        qml.CRX(weights_cross_crx[i + qubits], wires=[i + qubits, i])

    # Cross CNOT 双向交叉
    for i in range(qubits):
        qml.CNOT(wires=[i, i + qubits])
        qml.CNOT(wires=[i + qubits, i])

    # 第二轮 Rot + CNOT（对 Q）
    for i in range(qubits):
        idx = i * 3
        qml.RZ(weights_cross_rot[idx], wires=i)
        qml.RX(weights_cross_rot[idx + 1], wires=i)
        qml.RZ(weights_cross_rot[idx + 2], wires=i)

    return [qml.expval(qml.PauliZ(wires=w)) for w in range(0, qubits)] + \
           [qml.expval(qml.PauliX(wires=w)) for w in range(0, qubits)]


# qmha_value
@qml.qnode(dev_value, interface="torch", diff_method="backprop")
def qmha_value(xv, score, weights_v_rot, weights_v_crx, gates=None):
    qml.AngleEmbedding(xv, wires=list(range(qubits)))
    initQKV_on_wires(weights_v_rot, weights_v_crx)

    # c= np.pi
    for i in range(qubits):
        # qml.RX(torch.tanh(score[i]) * c, wires=i)
        qml.RX(torch.tanh(score[i]) * gates[i], wires=i)

    # CNOT Ring
    for i in range(qubits):
        qml.CNOT(wires=[i, (i + 1) % qubits])

    return [qml.expval(qml.PauliZ(wires=w)) for w in range(qubits)] + \
           [qml.expval(qml.PauliX(wires=w)) for w in range(qubits)]
           

# qLatentAttention
class QuantumAttention(nn.Module):
    def __init__(self, in_embed=96):
        super().__init__()
        self.qubits = qubits
        embed_dim = qubits 
        self.pre_net = nn.Linear(in_embed, embed_dim)

        # Rot 参数 24 个
        self.weights_q_rot = nn.Parameter(torch.empty(qubits*6).uniform_(-np.pi / 2, np.pi / 2))
        self.weights_k_rot = nn.Parameter(torch.empty(qubits*6).uniform_(-np.pi / 2, np.pi / 2))
        self.weights_v_rot = nn.Parameter(torch.empty(qubits*6).uniform_(-np.pi / 2, np.pi / 2))
        self.weights_cross_rot = nn.Parameter(torch.empty(qubits*3).uniform_(-np.pi / 2, np.pi / 2))

        # CRX θ 参数 → 4 个，1 轮
        self.weights_q_crx = nn.Parameter(torch.randn(qubits*2) * 0.1)
        self.weights_k_crx = nn.Parameter(torch.randn(qubits*2) * 0.1)
        self.weights_v_crx = nn.Parameter(torch.randn(qubits*2) * 0.1)
        self.weights_cross_crx = nn.Parameter(torch.randn(qubits*2) * 0.1)

        self.score_gates = nn.Parameter(torch.ones(qubits) * np.pi)


        # expval 结果 4 + 4 = 8 per attention
        self.norm = nn.LayerNorm(24)
        mlp_hidden_dim = 4 * embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(24, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.5),  # 防止 bias collapse
            nn.Linear(mlp_hidden_dim, embed_dim)
        )
        self.dropout = nn.Dropout(0.5)

        self.linear = nn.Sequential(
            nn.LayerNorm(embed_dim*2),
            nn.ReLU(),
            # nn.Linear(embed_dim*2, 2)
        )

    def forward(self, x1, x2):
        x1 = self.pre_net(x1)
        x2 = self.pre_net(x2)

        outputs1 = []
        outputs2 = []
        # outputs3 = []

        for i in range(x1.shape[0]):
            # x1-x2 attention
            score_1 = torch.stack(qmha_score(x1[i], x2[i],
                                             self.weights_q_rot, self.weights_q_crx,
                                             self.weights_k_rot, self.weights_k_crx, 
                                             self.weights_cross_rot, self.weights_cross_crx))
            value_1 = torch.stack(qmha_value(x2[i], torch.sqrt(score_1[:qubits]**2 + score_1[qubits:]**2),
                                             self.weights_v_rot, self.weights_v_crx
                                             , gates=self.score_gates
                                             ))
            outputs1.append(value_1)

        out = torch.stack(outputs1).float()

        # out = self.linear(out)

        return out
