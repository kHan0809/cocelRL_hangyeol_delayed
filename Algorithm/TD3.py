from Model.Model import Actor, Critic
import torch
import torch.nn.functional as F
from Common.Utils import copy_weight, soft_update, hard_update
from Common.Buffer import ReplayMemory

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


        self.critic      = Critic(num_inputs, action_space.shape[0]).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.lr)
        self.critic_target = Critic(num_inputs, action_space.shape[0]).to(self.device)
        hard_update(self.critic_target, self.critic)

        self.buffer = ReplayMemory(self.buffer_size, args.seed)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update_parameters(self, batch_size, updates):

        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.buffer.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)


        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action_batch) * 0.2).clamp(-0.5, 0.5)
            next_action = (self.actor_target(next_state_batch) + noise).clamp(-1.,1.)

            # Compute critic loss
            # target_Q1, target_Q2   = self.critic_target(next_state_batch,next_action)

            target_Q1, target_Q2 = self.critic_target(torch.cat((next_state_batch, next_action), 1))
            minq = torch.min(target_Q1, target_Q2)
            target_Q = reward_batch + mask_batch*self.gamma*minq

        # Q1, Q2 = self.critic(state_batch, action_batch)
        Q1, Q2 = self.critic(torch.cat((state_batch, action_batch),1))

        critic_loss = F.mse_loss(Q1,target_Q) + F.mse_loss(Q2,target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if (updates%2)==0:
            # Compute actor loss
            pi = self.actor(state_batch)
            actor_loss = -self.critic.Q1(torch.cat((state_batch, pi),1)).mean()
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update tareget networks
            soft_update(self.critic_target, self.critic, self.tau)
            soft_update(self.actor_target,  self.actor, self.tau)

            return (critic_loss.item(), actor_loss.item())



        return critic_loss.item()


