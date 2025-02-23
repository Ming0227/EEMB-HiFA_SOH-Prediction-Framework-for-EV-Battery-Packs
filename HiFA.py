from torch import nn
import torch
import torch.nn.functional as F


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


class HiFA(nn.Module):
    def __init__(self, num_inputs_1, num_inputs_2, num_inputs_3):
        super(HiFA, self).__init__()

        self.gated_attention = GatedAttention(num_inputs_1)
        self.self_attention = SelfAttention(num_inputs_2)
        self.self_attention_3 = SelfAttention(num_inputs_3)
        self.fc_merge = nn.Linear(num_inputs_3, num_inputs_3)

    def forward(self, c1, c2, c3, c4,):
        # HiFA
        c1_c2, _ = self.gated_attention(torch.cat((c1, c2), dim=1))
        c1_c2_c3, _ = self.self_attention_3(torch.cat((c1_c2, c3), dim=1))
        merged = self.fc_merge(torch.cat((c1_c2_c3, c4), dim=1))

        # Self-attention for merged features
        output, _ = self.self_attention(merged)

        return output
