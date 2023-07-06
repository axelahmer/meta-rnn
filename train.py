from collections import deque
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import gymnasium as gym
from rnn_agent import RNNAgent
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm


def train_agent(agent, num_episodes=400, hidden_size=64, iters=2, gamma=0.99, intrinsic_reward_coef=0.1, lr=0.001, render=False):
    # Initialize device
    device = torch.device('cpu')

    # Create the environment
    env_name = 'CartPole-v1'
    env = gym.make(env_name)

    # Initialize the policy network and optimizer
    agent = agent(env.observation_space.shape[0], env.action_space.n, hidden_size, iters=iters).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=lr)

    # Initialize the lists to hold the rewards for all episodes
    all_episode_rewards = []
    all_episode_raw_rewards = []

    episode_return_deque = deque(maxlen=10)
    episode_raw_return_deque = deque(maxlen=10)

    progress_bar = tqdm(range(num_episodes), position=0, leave=True)

    for episode in progress_bar:  # Add tqdm progress bar
        # Switch to human render mode for every 100th episode
        if render and episode % 100 == 0:
            env = gym.make(env_name, render_mode='human')
        else:
            env = gym.make(env_name)

        state, _ = env.reset()
        episode_states, episode_actions, episode_rewards, episode_intrinsic_rewards, episode_raw_rewards = [], [], [], [], []

        while True:
            # Policy iteration
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            dist, soft_confs = agent(state_tensor)
            action = dist.sample()

            # Interaction with the environment
            next_state, reward, term, trunc, _ = env.step(action.item())

            # Calculating intrinsic reward
            n = soft_confs.size(0)
            a = (torch.arange(0, n).float() / n)
            intrinsic_reward = -(soft_confs.cpu() * a).sum().item() * intrinsic_reward_coef

            # print soft confs on 1 line

            # Storing episode data
            episode_states.append(state_tensor)
            episode_actions.append(action)
            episode_rewards.append(reward + intrinsic_reward)
            episode_intrinsic_rewards.append(intrinsic_reward)
            episode_raw_rewards.append(reward)

            # Transition to the next state
            state = next_state

            if term or trunc:
                break

        # Store the total episode reward and raw reward
        total_reward = sum(episode_rewards)
        total_raw_reward = sum(episode_raw_rewards)

        all_episode_rewards.append(total_reward)
        all_episode_raw_rewards.append(total_raw_reward)

        # Add total raw reward to the deque
        episode_raw_return_deque.append(total_raw_reward)
        episode_return_deque.append(total_reward)

        # Add the mean reward to the progress bar description
        progress_bar.set_description(f'Episode {episode} | Mean R: {np.mean(episode_return_deque):.2f} | Mean raw R: {np.mean(episode_raw_return_deque):.2f}')
        
        # Computing returns for each time step
        returns = []
        G = 0
        for r in reversed(episode_rewards):
            G = r + gamma * G
            returns.insert(0, G)

        # Normalizing the returns
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        # Updating the policy parameters
        optimizer.zero_grad()
        loss = 0
        for state, action, G in zip(episode_states, episode_actions, returns):
            dist, _ = agent(state)
            log_prob = dist.log_prob(action)
            loss += -log_prob * G.item()

        loss.backward()
        optimizer.step()

    return all_episode_rewards

# Define number of simulations and episodes
simulations = 5
num_episodes = 300

# Create df to store rewards
df = pd.DataFrame(columns=['Episode', 'Reward'])

# Run simulations
for i in range(simulations):
    print(f'Trial : {i+1}/5')
    # You can replace RNNAgent with your agent class
    episode_rewards = train_agent(RNNAgent, num_episodes=num_episodes)
    df = pd.concat([df,pd.DataFrame({'Episode': range(1, num_episodes+1), 'Reward': episode_rewards})])

sns.lineplot(x='Episode', y='Reward', data=df)
plt.show()
