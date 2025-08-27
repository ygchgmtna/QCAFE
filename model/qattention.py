# model/qattention.py
import torch
import torch.nn as nn
from torch import Tensor
from model.model import AbstractModel
from model.QAFv2 import QuantumAttention, QAFConfig

class QAttention(AbstractModel):
    def __init__(self, in_dim=64, h_dim=64, num_classes=2):
        super(QAttention, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # 编码器
        self.encoder = _TextImageEncoder(shared_image_dim=in_dim,
                                         shared_text_dim=in_dim)

        # 跨模态量子注意力
        self.cross_modal = QuantumAttention(in_embed=64, 
                                            config=QAFConfig(use_two_ring=True, 
                                                             use_shifted_cross=True, 
                                                             use_anneal=True)).to(self.device)

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


class _TextImageEncoder(nn.Module):
    """
    encoding for input text and image
    """
    def __init__(self,
                 cnn_channel=32,
                 cnn_kernel_size=(1, 2, 4, 8),
                 shared_image_dim=128,
                 shared_text_dim=128):
        """
        Args:
            cnn_channel (int): the number of cnn channel, default=32
            cnn_kernel_size (int): the size of cnn kernel, default=(1, 2, 4, 8)
            shared_image_dim: output dim of image data
            shared_text_dim: output dim of text data
        """
        super(_TextImageEncoder, self).__init__()
        self.shared_text_encoding = _FastTextCNN(channel=cnn_channel,
                                                 kernel_size=cnn_kernel_size)
        self.shared_text_linear = nn.Sequential(
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(),
            nn.Linear(64, shared_text_dim), nn.BatchNorm1d(shared_text_dim),
            nn.ReLU())
        self.shared_image = nn.Sequential(
                                          nn.Linear(512, 256),
                                          nn.BatchNorm1d(256), nn.ReLU(),
                                          nn.Dropout(),
                                          nn.Linear(256, shared_image_dim),
                                          nn.BatchNorm1d(shared_image_dim),
                                          nn.ReLU())

    def forward(self, text: Tensor, image: Tensor):
        """
        Args:
            text (Tensor): batch text data, shape=(batch_size, 30, 200)
            image (Tensor): batch image data, shape=(batch_size, 512)
        Returns:
            text_shared (Tensor): Encoding text data, shape=(batch_size, 128)
            image_shared (Tensor): Encoding image data, shape=(batch_size, 128)
        """
        text_encoding = self.shared_text_encoding(text)
        text_shared = self.shared_text_linear(text_encoding)
        image_shared = self.shared_image(image)
        return text_shared, image_shared


class _FastTextCNN(nn.Module):
    # a CNN-based altertative approach of bert for text encoding
    def __init__(self, channel=32, kernel_size=(1, 2, 4, 8)):
        """
        Args:
            channel (int): the number of conv channel, default=32
            kernel_size:  (int): the size of cnn kernel, default=(1, 2, 4, 8)
        """
        super(_FastTextCNN, self).__init__()
        self.fast_cnn = nn.ModuleList()
        for kernel in kernel_size:
            self.fast_cnn.append(
                nn.Sequential(nn.Conv1d(200, channel, kernel_size=kernel),
                              nn.BatchNorm1d(channel), nn.ReLU(),
                              nn.AdaptiveMaxPool1d(1)))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): processed text data, shape=(batch_size,30,200)
        Returns:
            Tensor: FastCNN data,shape=(batch_size,128)
        """
        #print(f'\nx.shape is {x.shape}\n')
        #print(f'\nx.shape is {x.shape}\n')
        x = x.permute(0, 2, 1)
        x_out = []
        for module in self.fast_cnn:
            if (x.shape[0] == 1):
                x_out.append(module(x).view(1, -1))
            else:
                x_out.append(module(x).squeeze())
        x_out = torch.cat(x_out, 1)
        return x_out