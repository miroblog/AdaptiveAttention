import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import torch
import torch.nn as nn
import torch.nn.functional as F

# refer to https://arxiv.org/pdf/1612.01887.pdf

HIDDEN_DIM = 512
N_IMAGE_DESCRIPTOR = 49
IMAGE_RAW_DIM = 2048
EMB_DIM = 512
N_VOCAB = 40 * 1000 # for caption; 60 * 1000 for hashtag

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AdaptiveAttention(nn.Module):
    def __init__(self):
        super(AdaptiveAttention, self).__init__()
        self.w_h = nn.Parameter(torch.randn(N_IMAGE_DESCRIPTOR))
        self.W_v = nn.Parameter(torch.randn(HIDDEN_DIM,N_IMAGE_DESCRIPTOR))
        self.W_g = nn.Parameter(torch.randn(HIDDEN_DIM,N_IMAGE_DESCRIPTOR))

        self.W_s = nn.Parameter(torch.randn(HIDDEN_DIM,N_IMAGE_DESCRIPTOR))

    # V; image descriptor , s_t : visual sentinel, h_t: output(hidden) of the lstm
    def forward(self, V, s_t, h_t):
        # eq (6)
        image_part = torch.matmul(V, self.W_v) # V -1, 49, 2048 / W_v 2048, 49/ output -1, 49, 49
        single_hidden_part = torch.matmul(h_t, self.W_g) # h_t -1, 512/ W_g 512, 49 output -1, 49
        dummy_1 = torch.ones(single_hidden_part.size(0), 1, N_IMAGE_DESCRIPTOR).to(device) # -1, 1, 49
        hidden_part = torch.bmm(single_hidden_part.unsqueeze(2), dummy_1) # -1, 49, 1 bmm -1, 1, 49 output -1, 49, 49
        z_t = torch.matmul(torch.tanh(image_part+hidden_part), self.w_h)

        # eq (7)
        alpha_t = F.softmax(z_t, dim=1)

        # eq(8)
        c_t = torch.sum(V * alpha_t.unsqueeze(2),dim=1)

        # eq(12)

        attention_vs = torch.matmul(torch.tanh(torch.matmul(s_t, self.W_s) + single_hidden_part), self.w_h)
        concatenates = torch.cat([z_t, attention_vs.unsqueeze(1)], dim=1)
        alpha_t_hat = F.softmax(concatenates, dim=1)

        # beta_t = alpha_t[k+1] , last element of the alpha_t
        beta_t = alpha_t_hat[:,-1:]
        # eq(11)
        c_t_hat = beta_t * s_t + (1-beta_t) * c_t
        return c_t_hat

class ExtendedLSTM(nn.Module):
    def __init__(self, input_size, output_size):
        super(ExtendedLSTM, self).__init__()
        self.lstm = nn.LSTMCell(input_size=input_size, hidden_size=output_size)
        self.sentinel = VisualSentinel()
        # x_t is [w_t; v_g]
    def forward(self, x_t, prev_hidden, prev_cell_state):
        curr_hidden , curr_cell = self.lstm(x_t, (prev_hidden, prev_cell_state))
        s_t = self.sentinel(x_t, prev_hidden, curr_cell)
        return s_t, curr_hidden, curr_cell

class VisualSentinel(nn.Module):
    def __init__(self):
        super(VisualSentinel, self).__init__()
        # since x_t is [w_t; v_g]
        self.W_x = nn.Parameter(torch.randn(HIDDEN_DIM+EMB_DIM, HIDDEN_DIM))
        self.W_h = nn.Parameter(torch.randn(HIDDEN_DIM, HIDDEN_DIM))

    # m_t is the lstm cell state
    def forward(self, x_t, prev_h_t, m_t):
        # eq (9) / (10)
        g_t = torch.sigmoid(torch.matmul(x_t, self.W_x) + torch.matmul(prev_h_t, self.W_h)) # output -1, 512, 512
        s_t = torch.mul(g_t, torch.tanh(m_t))
        return s_t

class AdaptiveAttentionLSTMNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(AdaptiveAttentionLSTMNetwork, self).__init__()
        self.attention = AdaptiveAttention()
        self.extended_lstm = ExtendedLSTM(input_size=input_size, output_size=output_size)
        self.mlp = nn.Linear(HIDDEN_DIM, N_VOCAB, bias=False)
    # x_t : [w_t; v_g] / V = [v_1, v_2, ... v_49]
    def forward(self, V, x_t, prev_hidden, prev_cell_state):
        s_t, curr_hidden, curr_cell = self.extended_lstm(x_t, prev_hidden, prev_cell_state)
        c_t_hat = self.attention(V, s_t, curr_hidden)
        output = F.softmax(self.mlp(c_t_hat + curr_hidden),dim=1)
        return output, curr_hidden, curr_cell


def main():
    BATCH_SIZE = 32
    V = torch.randn((BATCH_SIZE, N_IMAGE_DESCRIPTOR, HIDDEN_DIM)).to(device)
    x_t = torch.randn((BATCH_SIZE, HIDDEN_DIM+EMB_DIM)).to(device)
    h_0, c_0 = get_start_states(batch_size=BATCH_SIZE)
    model = AdaptiveAttentionLSTMNetwork(input_size=HIDDEN_DIM+EMB_DIM, output_size=HIDDEN_DIM)
    model.to(device)
    output, curr_hidden, curr_cell = model(V, x_t, h_0, c_0)
    print("test done ...")

def get_start_states(batch_size):
    hidden_dim = HIDDEN_DIM
    h0 = torch.zeros(batch_size, hidden_dim).to(device)
    c0 = torch.zeros(batch_size, hidden_dim).to(device)
    return h0, c0

if __name__ == '__main__':
    main()