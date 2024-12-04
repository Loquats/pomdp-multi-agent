from src.env_utils import *
from src.belief import *
from src.mdp import BeliefStateMDP


def randstep(mdp, dsf_belief, action):
    """
    Ch. 21.7, algorithm 21.11 randstep
    """
    state = dsf_belief.sample() # stochastic
    next_state = mdp.sample_next_state(state, action) # stochastic
    reward = mdp.state_reward(state, action) # deterministic
    observation = mdp.get_observation(action, next_state=next_state) # deterministic
    next_belief = update(dsf_belief, observation, action) # deterministic
    return next_belief, reward, observation

def rollout(mdp, observation, dsf_belief: DiscreteStateFilter, policy, depth):
    """
    the only stochastic part of this function is randstep
    """
    discounted_reward = 0.0
    prev_action = None
    first_action = None
    for i in range(depth):
        action = policy.get_action(observation, prev_action)
        if first_action is None:
            first_action = action
        dsf_belief, reward, observation = randstep(mdp, dsf_belief, action)
        discounted_reward += mdp.gamma**i * reward
        prev_action = action
        # print(f"depth {i} {action} {reward}")
    return first_action, discounted_reward


def greedy(mdp, utility, dsf_belief):
    # u, action = 
    # return (u, action)
    pass

def lookahead():
    pass

def utility_estimate(belief):
    """
    heuristic: variance of 2d grid, or number of zero cells, etc.
    """
    return 0