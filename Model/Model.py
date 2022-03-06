import torch
import torch.nn as nn
import torch.nn.functional as F
from Common.Utils import weight_init
from torch.distributions import Normal

class Actor(nn.Module):
    def __init__(self,state_dim, action_dim,hidden_dim=256):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(self.state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, self.action_dim)

    def forward(self, x):
        L1 = F.relu(self.fc1(x))
        L2 = F.relu(self.fc2(L1))
        output = torch.tanh(self.fc3(L2))
        return output



class Critic(nn.Module):
    def __init__(self,state_dim, action_dim):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(self.state_dim+self.action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

        self.q2_fc1 = nn.Linear(self.state_dim + self.action_dim, 400)
        self.q2_fc2 = nn.Linear(400, 300)
        self.q2_fc3 = nn.Linear(300, 1)

    def forward(self, x):
        L1 = F.relu(self.fc1(x))
        L2 = F.relu(self.fc2(L1))
        q1_output = self.fc3(L2)

        L1 = F.relu(self.q2_fc1(x))
        L2 = F.relu(self.q2_fc2(L1))
        q2_output = self.q2_fc3(L2)
        return q1_output, q2_output

    def Q1(self, x):
        L1 = F.relu(self.fc1(x))
        L2 = F.relu(self.fc2(L1))
        q1_output = self.fc3(L2)
        return q1_output

class State_Predict(nn.Module):
    def __init__(self, state_dim, action_dim, delay_len,device):
        super(State_Predict, self).__init__()
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.delay = delay_len
        self.os_dim     = state_dim

        self.GRU = nn.GRU(input_size=self.state_dim, hidden_size=self.state_dim, num_layers=1,batch_first=True)
        self.hidden0 = torch.randn(1,1,self.state_dim).to(device=device)
        #======predictive architecture=====
        self.N = nn.Linear(self.state_dim + self.action_dim + self.state_dim, self.state_dim)
        self.D = nn.Linear(self.state_dim + self.action_dim + self.state_dim, self.state_dim)
        self.F = nn.Linear(self.state_dim + self.action_dim + self.state_dim, self.state_dim)


    def forward(self,state,action_buffer):
        o, hidden = self.GRU(state, self.hidden0)
        for i in range(self.delay):
            state = self.Predict(state,torch.reshape(action_buffer[i],(1,1,-1)),o) # (to match state dim) #(batch, seqence, input size)
            o, hidden = self.GRU(state, hidden)
        return o

    def forward_batch(self,batch_size,state,action_buffer):
        hidden0 = self.hidden0.repeat(1,batch_size,1).to(device=state.device) #hidden0 size = (1,997,17) (1, batch, state_dim)
        state = state.unsqueeze(1)

        o, hidden = self.GRU(state, hidden0)
        for i in range(self.delay):
            state = self.Predict(state, action_buffer[:,i,:].unsqueeze(1), o)
            o, hidden = self.GRU(state, hidden)
        return o

    def forward_4_train(self,batch_size,state,action_delayed):
        hidden0 = self.hidden0.repeat(1, batch_size, 1).to(device=state.device)
        state = state.unsqueeze(1)
        o, hidden = self.GRU(state, hidden0)
        state = self.Predict(state, action_delayed, o)
        return state

    def Predict(self,state,action,o):
        D = self.D(torch.cat((state, action, o), 2))
        N = self.N(torch.cat((state, action, o), 2))
        F = self.F(torch.cat((state, action, o), 2))

        return F*(state+D)+(1-F)*N

