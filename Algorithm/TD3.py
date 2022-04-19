import numpy as np

from Model.Model import Actor, Critic4Delay, State_Predict
import torch
import torch.nn.functional as F
from Common.Utils import  soft_update, hard_update
from Common.Buffer import ReplayMemory
# torch.autograd.set_detect_anomaly(True)

class TD3():

    def __init__(self, num_inputs, action_space, args):

        #Control hyperparameters
        self.buffer_size = args.replay_size
        self.batch_size  = args.batch_size

        self.gamma = args.gamma
        self.tau = args.tau
        self.device = torch.device("cuda" if args.cuda else "cpu")


        self.actor      = Actor(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.lr)
        self.actor_target = Actor(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.actor_target, self.actor)


        self.critic      = Critic4Delay(num_inputs, action_space.shape[0],args.delay).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.lr)
        self.critic_target = Critic4Delay(num_inputs, action_space.shape[0],args.delay).to(self.device)
        hard_update(self.critic_target, self.critic)

        self.buffer            = ReplayMemory(self.buffer_size, args.seed)
        self.buffer4transition = ReplayMemory(self.buffer_size, args.seed)

        self.state_predict = State_Predict(num_inputs,action_space.shape[0],args.delay,self.device).to(self.device)
        self.GRU_optimizer = torch.optim.Adam(self.state_predict.GRU.parameters(), lr=args.lr)
        self.F_opt         = torch.optim.Adam(self.state_predict.F.parameters(), lr=args.lr)
        self.N_opt         = torch.optim.Adam(self.state_predict.N.parameters(), lr=args.lr)
        self.D_opt         = torch.optim.Adam(self.state_predict.D.parameters(), lr=args.lr)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def select_predict_action(self,state,action_buffer):
        state         = torch.FloatTensor(state).to(device = self.device)
        state         = torch.reshape(state, (1, 1, -1))

        action_buffer = np.array(action_buffer)
        action_buffer = torch.FloatTensor(action_buffer).to(device=self.device)
        o = self.state_predict(state, action_buffer)
        return self.actor(o).cpu().data.numpy().flatten()

    def update_parameters(self, batch_size, updates, delay):
        state_batch, action_buffer_batch, reward_batch, next_state_batch, mask_batch = self.buffer.sample(batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_buffer_batch = torch.FloatTensor(action_buffer_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            o,predict_next_state = self.state_predict.forward_batch(state_batch.shape[0],next_state_batch,action_buffer_batch[:,1:,:])
            next_action = self.actor_target(o)
            noise = (torch.randn_like(next_action) * 0.2).clamp(-0.5, 0.5)
            next_action = (next_action+noise).clamp(-1.,1.)

            next_action_buffer = torch.cat((action_buffer_batch[:,1:,:],next_action),1).reshape(batch_size,-1)
            target_Q1, target_Q2 = self.critic_target(torch.cat((next_state_batch, next_action_buffer), 1))

            minq = torch.min(target_Q1, target_Q2)
            target_Q = reward_batch + mask_batch*self.gamma*minq

        Q1, Q2 = self.critic(torch.cat((state_batch, action_buffer_batch.reshape(batch_size,-1)),1))

        critic_loss = F.mse_loss(Q1,target_Q) + F.mse_loss(Q2,target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


        #==================================actor train==========================
        if (updates%2)==0:
            # with torch.autograd.detect_anomaly():
                # Compute actor loss
            o,predict_state = self.state_predict.forward_batch(state_batch.shape[0], state_batch, action_buffer_batch[:,:-1,:])
            pi = self.actor(o)

            pi_action_buffer = torch.cat((action_buffer_batch[:,:-1,:], pi), 1).reshape(batch_size, -1)

            actor_loss = -self.critic.Q1(torch.cat((state_batch, pi_action_buffer),1)).mean()
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            self.GRU_optimizer.zero_grad()
            actor_loss.backward()
            self.GRU_optimizer.step()
            self.actor_optimizer.step()


            soft_update(self.critic_target, self.critic, self.tau)
            soft_update(self.actor_target,  self.actor, self.tau)

            return Q1.mean().item(), Q2.mean().item(), critic_loss.item(), actor_loss.item()


        return Q1.mean().item(), Q2.mean().item(), critic_loss.item(), critic_loss.item()

    def transition_update_parameters(self, delay):
        state_batch, action_buffer_batch, reward_batch, next_state_batch, mask_batch = self.buffer4transition.sample_all()

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_buffer_batch = torch.FloatTensor(action_buffer_batch).to(self.device)
        # ===============transition train==========================================
        o, predict_state = self.state_predict.forward_batch(state_batch[:-delay,:].shape[0], state_batch[:-delay,:], action_buffer_batch[:-delay, :-1, :])

        trans_loss = F.mse_loss(predict_state.squeeze(1), state_batch[delay:,:])

        self.F_opt.zero_grad()
        self.N_opt.zero_grad()
        self.D_opt.zero_grad()
        self.GRU_optimizer.zero_grad()
        trans_loss.backward()
        self.GRU_optimizer.step()
        self.F_opt.step()
        self.N_opt.step()
        self.D_opt.step()

        return trans_loss.item()
import numpy as np

from Model.Model import Actor, Critic4Delay, State_Predict
import torch
import torch.nn.functional as F
from Common.Utils import  soft_update, hard_update
from Common.Buffer import ReplayMemory
# torch.autograd.set_detect_anomaly(True)

class TD3():

    def __init__(self, num_inputs, action_space, args):

        #Control hyperparameters
        self.buffer_size = args.replay_size
        self.batch_size  = args.batch_size

        self.gamma = args.gamma
        self.tau = args.tau
        self.device = torch.device("cuda" if args.cuda else "cpu")


        self.actor      = Actor(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.lr)
        self.actor_target = Actor(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.actor_target, self.actor)


        self.critic      = Critic4Delay(num_inputs, action_space.shape[0],args.delay).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.lr)
        self.critic_target = Critic4Delay(num_inputs, action_space.shape[0],args.delay).to(self.device)
        hard_update(self.critic_target, self.critic)

        self.buffer            = ReplayMemory(self.buffer_size, args.seed)
        self.buffer4transition = ReplayMemory(self.buffer_size, args.seed)

        self.state_predict = State_Predict(num_inputs,action_space.shape[0],args.delay,self.device).to(self.device)
        self.GRU_optimizer = torch.optim.Adam(self.state_predict.GRU.parameters(), lr=args.lr)
        self.F_opt         = torch.optim.Adam(self.state_predict.F.parameters(), lr=args.lr)
        self.N_opt         = torch.optim.Adam(self.state_predict.N.parameters(), lr=args.lr)
        self.D_opt         = torch.optim.Adam(self.state_predict.D.parameters(), lr=args.lr)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def select_predict_action(self,state,action_buffer):
        state         = torch.FloatTensor(state).to(device = self.device)
        state         = torch.reshape(state, (1, 1, -1))

        action_buffer = np.array(action_buffer)
        action_buffer = torch.FloatTensor(action_buffer).to(device=self.device)
        o = self.state_predict(state, action_buffer)
        return self.actor(o).cpu().data.numpy().flatten()

    def update_parameters(self, batch_size, updates, delay):
        state_batch, action_buffer_batch, reward_batch, next_state_batch, mask_batch = self.buffer.sample(batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_buffer_batch = torch.FloatTensor(action_buffer_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            o,predict_next_state = self.state_predict.forward_batch(state_batch.shape[0],next_state_batch,action_buffer_batch[:,1:,:])
            next_action = self.actor_target(o)
            noise = (torch.randn_like(next_action) * 0.2).clamp(-0.5, 0.5)
            next_action = (next_action+noise).clamp(-1.,1.)

            next_action_buffer = torch.cat((action_buffer_batch[:,1:,:],next_action),1).reshape(batch_size,-1)
            target_Q1, target_Q2 = self.critic_target(torch.cat((next_state_batch, next_action_buffer), 1))

            minq = torch.min(target_Q1, target_Q2)
            target_Q = reward_batch + mask_batch*self.gamma*minq

        Q1, Q2 = self.critic(torch.cat((state_batch, action_buffer_batch.reshape(batch_size,-1)),1))

        critic_loss = F.mse_loss(Q1,target_Q) + F.mse_loss(Q2,target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


        #==================================actor train==========================
        if (updates%2)==0:
            # with torch.autograd.detect_anomaly():
                # Compute actor loss
            o,predict_state = self.state_predict.forward_batch(state_batch.shape[0], state_batch, action_buffer_batch[:,:-1,:])
            pi = self.actor(o)

            pi_action_buffer = torch.cat((action_buffer_batch[:,:-1,:], pi), 1).reshape(batch_size, -1)

            actor_loss = -self.critic.Q1(torch.cat((state_batch, pi_action_buffer),1)).mean()
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            self.GRU_optimizer.zero_grad()
            actor_loss.backward()
            self.GRU_optimizer.step()
            self.actor_optimizer.step()

            with torch.no_grad():
                soft_update(self.critic_target, self.critic, self.tau)
                soft_update(self.actor_target,  self.actor, self.tau)

            return Q1.mean().item(), Q2.mean().item(), critic_loss.item(), actor_loss.item()


        return Q1.mean().item(), Q2.mean().item(), critic_loss.item(), critic_loss.item()

    def transition_update_parameters(self, delay):
        state_batch, action_buffer_batch, reward_batch, next_state_batch, mask_batch = self.buffer4transition.sample_all()

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_buffer_batch = torch.FloatTensor(action_buffer_batch).to(self.device)
        # ===============transition train==========================================
        o, predict_state = self.state_predict.forward_batch(state_batch[:-delay,:].shape[0], state_batch[:-delay,:], action_buffer_batch[:-delay, :-1, :])

        trans_loss = F.mse_loss(predict_state.squeeze(1), state_batch[delay:,:])

        self.F_opt.zero_grad()
        self.N_opt.zero_grad()
        self.D_opt.zero_grad()
        self.GRU_optimizer.zero_grad()
        trans_loss.backward()
        self.GRU_optimizer.step()
        self.F_opt.step()
        self.N_opt.step()
        self.D_opt.step()

        return trans_loss.item()
