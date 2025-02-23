from tcn import TemporalConvNet
from torch import nn
import torch
from transfer_losses import TransferLoss
import torch.nn.functional as F

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


# TCN net to extract the V, C, Q separately
class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y1 = self.tcn(x)  # [64, 16, 200]
        # print(y1.shape)
        return self.linear(y1[:, :, -1])  # [64, output_size])


class multi_TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout, fc_output):
        super(multi_TCN, self).__init__()
        self.net = TCN(input_size, output_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.fc = nn.Linear(3 * output_size, fc_output)
        self.init_weights()

    def init_weights(self):
        self.fc.weight.data.normal_(0, 0.01)

    def forward(self, x1, x2, x3):
        x1 = self.net(x1)  # [batch_size, output_size]
        x2 = self.net(x2)
        x3 = self.net(x3)
        x = torch.cat([x1, x2, x3], -1)  # [batch_size, output_size*3]
        x_fc = self.fc(x)  # [batch_size, fc_output]
        return x_fc


# 2d-CNN net to extract from the V, C, Q images
class LeNet5(nn.Module):
    def __init__(self, in_channels, out_channels, features, output_size):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, out_channels, 5),
            nn.ReLU(),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d(features)
        self.fc1 = nn.Linear(features ** 2 * out_channels, output_size)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.adaptive_pool(y)
        y = y.view(y.shape[0], -1)
        outputs = self.fc1(y)
        return outputs


class multi_LeNet5(nn.Module):
    def __init__(self, in_channels, out_channels, features, output_size):
        super(multi_LeNet5, self).__init__()
        self.net = LeNet5(in_channels, out_channels, features, output_size)

    def forward(self, x1, x2, x3):
        x = torch.stack([x1, x2, x3], dim=1)
        x = self.net(x)
        return x


# Shallow Conv1d for the IC-based
class Conv1d(nn.Module):
    def __init__(self, input_size, output_size):
        super(Conv1d, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=self.input_size, out_channels=128, kernel_size=5, stride=1, bias=False,
                      padding=(5 // 2)),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.1)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(128, 64,
                      kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, 32,
                      kernel_size=4, stride=1, bias=False, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool1d(10)

        self.fc = nn.Linear(10 * 32, output_size)

    def forward(self, x):
        x_in = x
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.adaptive_pool(x)
        x = x.reshape(x.shape[0], -1)
        x_fc = self.fc(x)
        return x_fc


class multi_Conv1d(nn.Module):
    def __init__(self, input_size, output_size):
        super(multi_Conv1d, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.CNN = Conv1d(self.input_size, self.output_size)
        self.fc = nn.Linear(3 * self.output_size, self.output_size)

    def forward(self, x1, x2, x3):
        x1 = self.CNN(x1)
        x2 = self.CNN(x2)
        x3 = self.CNN(x3)
        x = torch.cat([x1, x2, x3], -1)
        x = self.fc(x)
        return x


# LSTM net to SOH
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2, noise_factor=0.01):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.noise_factor = noise_factor
        self.num_directions = 1  # 单向LSTM
        self.dropout = dropout
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq, batch_size=None):
        if batch_size is None:
            batch_size = input_seq.shape[0]

        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)

        output, _ = self.lstm(input_seq, (h_0, c_0))  # output(batch_size, seq_len, num_directions * hidden_size)
        pred = self.linear(output)  # (batch_size, seq_len, output_size)

        return pred[:, -1, :]


class MLP(nn.Module):
    def __init__(self, num_inputs, num_outputs, drop_out=0.2):
        super().__init__()
        self.dropout = nn.Dropout(drop_out)

        ## MLP network
        self.MLP = nn.Sequential(nn.Linear(num_inputs, 128),
                                 nn.ReLU(), self.dropout,
                                 nn.Linear(128, 64),
                                 nn.ReLU(), self.dropout)

        self.linear = nn.Linear(64, num_outputs)

    def forward(self, x):
        x = x.squeeze()
        c = self.MLP(x)
        c = self.linear(c)

        return c


# self-attention class
class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.projected_dim = input_dim // 2

        self.fc1 = nn.Linear(input_dim, self.projected_dim)

        self.fc1_output_dim = self.projected_dim

        self.fc2 = nn.Linear(self.projected_dim, input_dim)

        self.fc2_output_dim = input_dim

    def forward(self, x):
        projected = self.fc1(x)
        weights = torch.softmax(self.fc2(projected), dim=1)
        weighted_input = x * weights

        return weighted_input, weights


# GatedAttention
class GatedAttention(nn.Module):
    def __init__(self, input_dim):
        super(GatedAttention, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention_gate = self.gate(x)
        gated_input = x * attention_gate

        return gated_input, attention_gate


# EEMB-HiFA
class MHMB(nn.Module):
    def __init__(self, tcn_input_size, tcn_output_size, tcn_num_channels, tcn_kernel_size, tcn_dropout, tcn_fc_output,
                 len_in_channels, len_out_channels, len_features, len_output_size, cnn_input_size, cnn_output_size,
                 lstm_input_size, lstm_hidden_size, lstm_num_layers, lstm_output_size, num_inputs, num_hidden1,
                 num_hidden2,
                 num_hidden3, num_outputs, drop_out):
        super(MHMB, self).__init__()

        self.tcn = multi_TCN(tcn_input_size, tcn_output_size, tcn_num_channels, tcn_kernel_size, tcn_dropout,
                             tcn_fc_output)
        self.LeNet5 = multi_LeNet5(len_in_channels, len_out_channels, len_features, len_output_size)
        self.cnn = multi_Conv1d(cnn_input_size, cnn_output_size)  # Conv1d
        self.lstm = LSTM(lstm_input_size, lstm_hidden_size, lstm_num_layers, lstm_output_size)
        self.MLP1 = MLP(150, lstm_output_size)
        self.dropout = nn.Dropout(drop_out)
        self.self_attention = SelfAttention(num_inputs)
        self.gated_attention = GatedAttention(100)
        self.self_attention_3 = SelfAttention(150)
        # Linear layer for merging c1, c2, c3, and c4
        self.fc_merge = nn.Linear(200, 200)

        ## MLP network
        self.MLP = nn.Sequential(nn.Linear(num_inputs, num_hidden1),
                                 nn.ReLU(), self.dropout,
                                 nn.Linear(num_hidden1, num_hidden2),
                                 nn.ReLU(), self.dropout,
                                 nn.Linear(num_hidden2, num_hidden3),
                                 nn.ReLU(), self.dropout)

        self.linear = nn.Linear(num_hidden3, num_outputs)

    def forward(self, x1, x2, x3, x4, x5, x6, x7):
        c1 = self.tcn(x1, x2, x3)
        c2 = self.LeNet5(x1, x2, x3)
        c3 = self.cnn(x4, x5, x6)
        c4 = self.lstm(x7)

        # HiFA
        c1_c2, _ = self.gated_attention(torch.cat((c1, c2), dim=1))
        c1_c2_c3, _ = self.self_attention_3(torch.cat((c1_c2, c3), dim=1))
        merged = self.fc_merge(torch.cat((c1_c2_c3, c4), dim=1))

        # Self-attention for merged features
        output, _ = self.self_attention(merged)

        c_mlp = self.MLP(output)
        y_predicted = self.linear(c_mlp)

        return y_predicted


if __name__ == '__main__':
    x_dim = 150
    x1_dim = 200
    y_dim = 1

    batch_size = 64

    tcn_input_size = 150
    tcn_output_size = 50
    tcn_num_channels = [256, 128, 64, 32, 16]
    tcn_kernel_size = 3
    tcn_dropout = 0.2
    tcn_fc_output = 50

    len_in_channels = 3
    len_out_channels = 32
    len_features = 50
    len_output_size = 50

    cnn_input_size = 150
    cnn_output_size = 50

    lstm_input_size = 1
    lstm_hidden_size = 64
    lstm_num_layers = 1
    lstm_output_size = 50

    num_inputs = 200
    num_hidden1 = 128
    num_hidden2 = 64
    num_hidden3 = 32
    num_outputs = 1

    drop_out = 0.2

    window_size = 20

    x1 = torch.randn(batch_size, x_dim, x1_dim).to(device)
    x2 = torch.randn(batch_size, x_dim, x1_dim).to(device)
    x3 = torch.randn(batch_size, x_dim, x1_dim).to(device)
    x4 = torch.randn(batch_size, x_dim, x1_dim).to(device)
    x5 = torch.randn(batch_size, x_dim, x1_dim).to(device)
    x6 = torch.randn(batch_size, x_dim, x1_dim).to(device)
    x7 = torch.randn(batch_size, x_dim, 1).to(device)

    y = torch.randn(batch_size, y_dim).to(device)

    MHMB = MHMB(tcn_input_size, tcn_output_size, tcn_num_channels, tcn_kernel_size, tcn_dropout, tcn_fc_output,
                len_in_channels, len_out_channels, len_features, len_output_size, cnn_input_size, cnn_output_size,
                lstm_input_size, lstm_hidden_size, lstm_num_layers, lstm_output_size, num_inputs, num_hidden1,
                num_hidden2, num_hidden3, num_outputs, drop_out).to(device)

    c_mlp, y_predicted = MHMB(x1, x2, x3, x4, x5, x6, x7)
    print('predict', y_predicted)
