# Rot+CRX(1轮)+CNOT
# CRX 只用 4 个参数（每 qubit 1 个）；
# 第二轮只做 Rot + CNOT Ring；


import pennylane as qml
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 定义 score 和 value 用的 device
dev_score = qml.device("default.qubit", wires=8, shots=None)
dev_value = qml.device("default.qubit", wires=4, shots=None)

def initQKV_on_wires(rot_params, crx_params, offset):
    # 每个 qubit 3 个参数：Z–X–Z，共 4 个 qubit，需要 12 个参数
    def zxz(idx, wire):
        qml.RZ(rot_params[idx], wires=wire)
        qml.RX(rot_params[idx + 1], wires=wire)
        qml.RZ(rot_params[idx + 2], wires=wire)

    # 第 1 层 ZXZ + CRX + CNOT
    zxz(0, offset + 0)
    zxz(3, offset + 1)
    zxz(6, offset + 2)
    zxz(9, offset + 3)

    qml.CRX(crx_params[0], wires=[offset + 0, offset + 1])
    qml.CRX(crx_params[1], wires=[offset + 1, offset + 2])
    qml.CRX(crx_params[2], wires=[offset + 2, offset + 3])
    qml.CRX(crx_params[3], wires=[offset + 3, offset + 0])

    qml.CRX(crx_params[4], wires=[offset + 0, offset + 3])
    qml.CRX(crx_params[5], wires=[offset + 1, offset + 0])
    qml.CRX(crx_params[6], wires=[offset + 2, offset + 1])
    qml.CRX(crx_params[7], wires=[offset + 3, offset + 2])
    
    qml.CNOT(wires=[offset + 0, offset + 1])  
    qml.CNOT(wires=[offset + 1, offset + 2])     
    qml.CNOT(wires=[offset + 2, offset + 3])  
    qml.CNOT(wires=[offset + 3, offset + 0])

    # 第 2 层 ZXZ + CRX + CNOT Ring    
    qml.CNOT(wires=[offset + 0, offset + 3])
    qml.CNOT(wires=[offset + 1, offset + 0])
    qml.CNOT(wires=[offset + 2, offset + 1])
    qml.CNOT(wires=[offset + 3, offset + 2])

    zxz(12, offset + 0)
    zxz(15, offset + 1)
    zxz(18, offset + 2)
    zxz(21, offset + 3)


# qmha_score
@qml.qnode(dev_score, interface="torch", diff_method="backprop")
def qmha_score(xq, xk, weights_q_rot, weights_q_crx, weights_k_rot, weights_k_crx):

    qml.AngleEmbedding(xq, wires=[0, 1, 2, 3])
    initQKV_on_wires(weights_q_rot, weights_q_crx, offset=0)

    qml.AngleEmbedding(xk, wires=[0, 1, 2, 3])
    initQKV_on_wires(weights_k_rot, weights_k_crx, offset=4)

    qml.CNOT(wires=[0, 4])
    qml.CNOT(wires=[1, 5])
    qml.CNOT(wires=[2, 6])
    qml.CNOT(wires=[3, 7])

    return [qml.expval(qml.PauliZ(wires=w)) for w in [4,5,6,7]] + \
           [qml.expval(qml.PauliX(wires=w)) for w in [4,5,6,7]]

# qmha_value
@qml.qnode(dev_value, interface="torch", diff_method="backprop")
def qmha_value(xv, score, weights_v_rot, weights_v_crx):

    qml.AngleEmbedding(xv, wires=[0, 1, 2, 3])
    initQKV_on_wires(weights_v_rot, weights_v_crx, offset=0)

    c = np.pi
    for i, w in enumerate([0, 1, 2, 3]):
        # qml.RZ(score[i], wires=w)
        qml.RX(torch.tanh(score[i]) * c, wires=w)

    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[3, 0])

    return [qml.expval(qml.PauliZ(wires=w)) for w in [0,1,2,3]] + \
           [qml.expval(qml.PauliX(wires=w)) for w in [0,1,2,3]]

# qLatentAttention
class QuantumAttention(nn.Module):
    def __init__(self, embed_dim=4, in_embed=96):
        super().__init__()

        self.pre_net = nn.Linear(in_embed, embed_dim)

        # Rot 参数 24 个
        self.weights_q_rot1 = nn.Parameter(torch.empty(24).uniform_(-np.pi / 2, np.pi / 2))
        self.weights_k_rot1 = nn.Parameter(torch.empty(24).uniform_(-np.pi / 2, np.pi / 2))
        self.weights_v_rot1 = nn.Parameter(torch.empty(24).uniform_(-np.pi / 2, np.pi / 2))

        self.weights_q_rot2 = nn.Parameter(torch.empty(24).uniform_(-np.pi / 2, np.pi / 2))
        self.weights_k_rot2 = nn.Parameter(torch.empty(24).uniform_(-np.pi / 2, np.pi / 2))
        self.weights_v_rot2 = nn.Parameter(torch.empty(24).uniform_(-np.pi / 2, np.pi / 2))

        self.weights_q_rot3 = nn.Parameter(torch.empty(24).uniform_(-np.pi / 2, np.pi / 2))
        self.weights_k_rot3 = nn.Parameter(torch.empty(24).uniform_(-np.pi / 2, np.pi / 2))
        self.weights_v_rot3 = nn.Parameter(torch.empty(24).uniform_(-np.pi / 2, np.pi / 2))

        # CRX θ 参数 → 4 个，1 轮
        self.weights_q_crx1 = nn.Parameter(torch.randn(8) * 0.1)
        self.weights_k_crx1 = nn.Parameter(torch.randn(8) * 0.1)
        self.weights_v_crx1 = nn.Parameter(torch.randn(8) * 0.1)

        self.weights_q_crx2 = nn.Parameter(torch.randn(8) * 0.1)
        self.weights_k_crx2 = nn.Parameter(torch.randn(8) * 0.1)
        self.weights_v_crx2 = nn.Parameter(torch.randn(8) * 0.1)

        self.weights_q_crx3 = nn.Parameter(torch.randn(8) * 0.1)
        self.weights_k_crx3 = nn.Parameter(torch.randn(8) * 0.1)
        self.weights_v_crx3 = nn.Parameter(torch.randn(8) * 0.1)

        # expval 结果 4 + 4 = 8 per attention, 3 attention → 24
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
                                             self.weights_q_rot1, self.weights_q_crx1,
                                             self.weights_k_rot1, self.weights_k_crx1))
            
            
            value_1 = torch.stack(qmha_value(x2[i], torch.sqrt(score_1[:4]**2 + score_1[4:]**2),
                                             self.weights_v_rot1, self.weights_v_crx1))

            outputs1.append(value_1)

            # x2-x3 attention
            # score_2 = torch.stack(qmha_score(x2[i], x3[i],
            #                                  self.weights_q_rot2, self.weights_q_crx2,
            #                                  self.weights_k_rot2, self.weights_k_crx2))

            # value_2 = torch.stack(qmha_value(x3[i], torch.sqrt(score_2[:4]**2 + score_2[4:]**2),
            #                                  self.weights_v_rot2, self.weights_v_crx2))

            # outputs2.append(value_2)

            # x1-x3 attention
            # score_3 = torch.stack(qmha_score(x1[i], x3[i],
            #                                  self.weights_q_rot3, self.weights_q_crx3,
            #                                  self.weights_k_rot3, self.weights_k_crx3))

            # value_3 = torch.stack(qmha_value(x3[i], torch.sqrt(score_3[:4]**2 + score_3[4:]**2),
            #                                  self.weights_v_rot3, self.weights_v_crx3))

            # outputs3.append(value_3)

        outputs1 = torch.stack(outputs1).float()
        # outputs2 = torch.stack(outputs2).float()
        # outputs3 = torch.stack(outputs3).float()


        # outputs = torch.cat([outputs1, outputs2, outputs3], dim=-1)

        # out = self.norm(outputs1)
        # out = self.dropout(out)
        # out = self.mlp(out)
        out = self.linear(outputs1)
        # print(f"out shape: {out.shape}, out dtype: {out.dtype}")
        return out
