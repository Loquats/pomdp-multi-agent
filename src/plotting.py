import os
import torch

import matplotlib
import matplotlib.pyplot as plt

from src.dqn import is_databricks_cluster

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
print(f"is_ipython: {is_ipython}")

if not is_databricks_cluster():
    plt.ion()

def plot_rewards(episode_rewards, episode_eps, show_result=False, save_dir=None):
    plt.figure("rewards")
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    eps_t = 100*torch.tensor(episode_eps, dtype=torch.float)
    if show_result:
        plt.title('Training Discounted Rewards')
    else:
        plt.clf()
        plt.title('Training Discounted Rewards...')
    plt.xlabel('Episode')
    plt.ylabel('Discounted Reward')
    plt.plot(rewards_t.numpy())
    plt.plot(eps_t.numpy())
    # Take 100 episode averages and plot them too
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    if is_databricks_cluster():
        pass
    else:
        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"rewards.png"))


def plot_loss(episode_loss, show_result=False, save_dir=None):
    plt.figure("loss")
    rewards_t = torch.tensor(episode_loss, dtype=torch.float)
    title = 'Training Loss (averaged per episode)'
    if show_result:
        plt.title(title)
    else:
        plt.clf()
        plt.title(f"{title}...")
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.plot(rewards_t.numpy())
    # Take 100 episode averages and plot them too
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    if is_databricks_cluster():
        pass
    else:
        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"loss.png"))