import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# 定義模型
class LSTMResNet(nn.Module):
    def __init__(self, num_classes, hidden_size, num_layers, dropout_rate=0.5):
        super(LSTMResNet, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = resnet.fc.in_features
        
        # LSTM 層，加入 Dropout
        self.lstm = nn.LSTM(input_size=self.feature_dim, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout_rate)
        
        # Dropout 層，用於全連接層前
        self.dropout = nn.Dropout(dropout_rate)
        
        # 最後的全連接分類層
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)
        # 提取特徵
        with torch.no_grad():
            features = self.feature_extractor(x)
        features = features.view(batch_size, seq_len, -1)
        
        # LSTM 運算
        lstm_out, _ = self.lstm(features)
        
        # 使用最後一個時間步的輸出
        final_output = lstm_out[:, -1, :]
        
        # Dropout 後進入分類層
        final_output = self.dropout(final_output)
        output = self.classifier(final_output)
        return output
