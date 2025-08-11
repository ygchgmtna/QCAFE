# model/qattention.py
import torch
import torch.nn as nn
from model.QA_flexible import QuantumAttention
from model.cafe import _TextImageEncoder

class QAttention(nn.Module):
    def __init__(self, in_dim=64, h_dim=64, num_classes=2):
        super(QAttention, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # 编码器
        self.encoder = _TextImageEncoder(shared_image_dim=in_dim,
                                         shared_text_dim=in_dim)

        # 跨模态量子注意力
        self.cross_modal = QuantumAttention(in_embed=in_dim).to(self.device)

        # 自动推断 cross_modal 输出维度
        with torch.no_grad():
            dummy_text = torch.zeros(1, in_dim, device=self.device)
            dummy_image = torch.zeros(1, in_dim, device=self.device)
            out_feat = self.cross_modal(dummy_text, dummy_image)
            out_dim = out_feat.shape[1]

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, h_dim),
            nn.BatchNorm1d(h_dim), nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.BatchNorm1d(h_dim), nn.ReLU(),
            nn.Linear(h_dim, num_classes)
        )

        self.loss_func_detection = nn.CrossEntropyLoss()

    def forward(self, text, image):
        text_encoded, image_encoded = self.encoder(text, image)
        fused_feat = self.cross_modal(text_encoded, image_encoded)
        return self.classifier(fused_feat)

    def calculate_loss(self, data):
        preds = self.forward(data['text'], data['image'])
        return self.loss_func_detection(preds, data['label'])

    def predict(self, data):
        out = self.forward(data['text'], data['image'])
        return torch.softmax(out, dim=-1)
