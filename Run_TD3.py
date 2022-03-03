import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from Algorithm.TD3 import TD3
from Common.Utils import set_seed, Eval, log_start, log_write

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="InvertedPendulumSwingDoubleLength-v2",help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Actor",help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',help='Automaically adjust α (default: False)')
parser.add_argument('--eval', type=bool, default=True,help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',help='learning rate (default: 0.0003)')
parser.add_argument('--seed', type=int, default=-1, metavar='N',help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=300001, metavar='N',help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=3000, metavar='N',help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', type=bool, default=True, help='run on CUDA (default: False)')
parser.add_argument('--log', default=True, type=bool, help='use tensorboard summary writer to log, if false, cannot use the features below')
args = parser.parse_args()

# log
for iteration in range(3,6):
    log_start("TD3_double_",iteration,log_flag=True)
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

        while not done:
            if args.start_steps > total_numsteps:
                action = env.action_space.sample() / action_limit
            else:
                action = (agent.select_action(state) + 0.1 * np.random.normal(0.0, 1.0, [env.action_space.shape[0]])).clip(-1.0,1.0)

            if len(agent.buffer) > args.batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    loss = agent.update_parameters(args.batch_size, updates)

                    updates += 1

            next_state, reward, done, _ = env.step(action * action_limit)  # Step
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward


            mask = 1 if episode_steps == env._max_episode_steps else float(not done)
            agent.buffer.push(state, action, reward, next_state, mask)  # Append transition to memory
            state = next_state

            #========================TEST or EVAL========================
            if (total_numsteps)%1000==0:
                Min_test_return, Avg_test_return, Max_test_return = Eval(test_env, agent, 10, False)
                print("----------------------------------------")
                print("Test Episodes: {}, Min. Return:{:.2f} Avg. Return: {:.2f} Max Return: {:.2f}".format(total_numsteps,Min_test_return,Avg_test_return,Max_test_return))
                print("----------------------------------------")
                log_write("TD3_double_", iteration, log_flag=True,total_step=total_numsteps,result=[Min_test_return,Avg_test_return,Max_test_return])
                torch.save(agent.actor.state_dict() ,'./model_save/actor_double.pth')
                torch.save(agent.critic.state_dict(), './model_save/critic_double.pth')


        if total_numsteps > args.num_steps:
            break
        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps,episode_steps,episode_reward))

    env.close()

