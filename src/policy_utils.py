from src.env_utils import *
from src.belief import *
from src.mdp import BeliefStateMDP


def randstep(mdp, dsf_belief, action):
    """
    Ch. 21.7, algorithm 21.11 randstep
    """
    state = dsf_belief.sample()
    next_state = mdp.sample_next_state(state, action)
    reward = mdp.state_reward(state, action)
    observation = mdp.get_observation(action, next_state=next_state)
    next_belief = update(dsf_belief, observation, action)
    return next_belief, reward, observation

# def lookahead_with_rollouts(mdp: BeliefStateMDP, observation: Observation, dsf_belief: DiscreteStateFilter, rollout_policy, depth: int):
#     """
#     Algorithm 9.1
#     See Chapter 22.1 for POMDPs, which is based on Ch. 9.2 for MDPs
#     This is a type of receding horizon planning (Ch. 9.1)

#     state in MDPs = dsf_belief in POMDPs
#     """
#     utility = rollout(mdp, observation, dsf_belief, rollout_policy, depth)
#     return greedy(mdp, 0, dsf_belief)

def rollout(mdp, observation, dsf_belief: DiscreteStateFilter, policy, depth):
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