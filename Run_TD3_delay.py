import argparse
import gym
import numpy as np
import itertools
import torch
from Algorithm.TD3 import TD3
from Common.Utils import set_seed, Eval, log_start, log_write, Action_Delay, Eval_delay
import copy
from collections import deque

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="HalfCheetah-v2",help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Actor",help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',help='Automaically adjust α (default: False)')
parser.add_argument('--eval', type=bool, default=True,help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.001, metavar='G',help='learning rate (default: 0.0003)')
parser.add_argument('--seed', type=int, default=-1, metavar='N',help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',help='hidden size (default: 256)')
parser.add_argument('--update_start_steps', type=int, default=1000, metavar='N',help='Steps sampling random actions (default: 10000)')
parser.add_argument('--update_every', type=int, default=50, metavar='N',help='update every 50(default) step')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',help='Steps sampling random actions (default: 10000)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', type=bool, default=True, help='run on CUDA (default: False)')
parser.add_argument('--log', default=True, type=bool, help='use tensorboard summary writer to log, if false, cannot use the features below')

parser.add_argument('--delay', type=int, default=2, help='Value target update per no. of updates per step (default: 1)')
args = parser.parse_args()

# log
def main(iteration):
    log_start("TD3_delay_",iteration,log_flag=True,dir="./Result_save/")
    args.seed=set_seed(args.seed)
    args.seed = args.seed + 1
    print("SEED : ", args.seed)

    # Environment
    # env = NormalizedActions(gym.make(args.env_name))
    env,test_env = gym.make(args.env_name), gym.make(args.env_name)
    print('env:', args.env_name, 'is created!')
    #==========seed related==========
    env.seed(args.seed), test_env.seed(args.seed)
    env.action_space.seed(args.seed), test_env.action_space.seed(args.seed)
    action_limit = env.action_space.high[0]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)


    # Agent
    agent = TD3(env.observation_space.shape[0], env.action_space, args)
    print('agent is created!')

    # Training Loop
    total_numsteps = 0
    updates = 0

    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        done = False

        state = env.reset()
        agent.buffer4transition.__init__(args.replay_size,args.seed)
        action_buffer = Action_Delay(env.action_space.shape[0], args.delay)

        while not done:
            # env.render()
            if args.start_steps > total_numsteps:
                action = env.action_space.sample() / action_limit
                action_buffer.append(action)
            else:
                action = (agent.select_predict_action(state,action_buffer.queue) + 0.1 * np.random.normal(0.0, 1.0, [env.action_space.shape[0]])).clip(-1.0,1.0)
                action_buffer.append(action)

            if total_numsteps >= args.update_start_steps and total_numsteps % args.update_every == 0 and len(agent.buffer.buffer) > args.batch_size:
                for j in range(args.update_every):
                    critic1_value, critic2_value, critic_loss, pi_loss = agent.update_parameters(args.batch_size, updates, args.delay)
                    updates += 1

            next_state, reward, done, _ = env.step(action_buffer.action() * action_limit)  # Step

            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            mask = 1 if episode_steps == env._max_episode_steps else float(not done)

            agent.buffer.push(state, copy.deepcopy(action_buffer.queue), reward, next_state, mask)
            agent.buffer4transition.push(state, copy.deepcopy(action_buffer.queue), reward, next_state, mask)

            state = next_state
            action_buffer.pop()
            #========================TEST or EVAL========================
            if (total_numsteps)%5000==0:
                Eval_action_buffer = Action_Delay(env.action_space.shape[0], args.delay)
                Min_test_return, Avg_test_return, Max_test_return = Eval_delay(test_env, agent, Eval_action_buffer, 3, False)
                print("----------------------------------------")
                print("Test Episodes: {}, Min. Return:{:.2f} Avg. Return: {:.2f} Max Return: {:.2f}".format(total_numsteps,Min_test_return.item(),Avg_test_return.item(),Max_test_return.item()))
                print("----------------------------------------")
                log_write("TD3_delay_", iteration, log_flag=True,total_step=total_numsteps,result=[Min_test_return,Avg_test_return,Max_test_return],dir="./Result_save/")
                # torch.save(agent.actor.state_dict() ,'./model_save/actor_double.pth')
                # torch.save(agent.critic.state_dict(), './model_save/critic_double.pth')

        if total_numsteps > args.batch_size:
            # Number of updates per step in environment
            for i in range(100):
                transition_loss = agent.transition_update_parameters(args.delay)
            print("Transition loss : {:.2f}".format(transition_loss))

        if total_numsteps > args.num_steps:
            break
        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps,episode_steps,episode_reward))
        if total_numsteps > args.update_start_steps:
            print("Critic1        : {:.4f}".format(critic1_value))
            print("Critic2        : {:.4f}".format(critic2_value))
            print("Critic loss    : {:.4f}".format(critic_loss))
            print("Policy loss    : {:.4f}".format(pi_loss))

    env.close()

if __name__ == '__main__':
    for iteration in range(2, 3):
        main(iteration)