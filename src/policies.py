import random
from src.dqn import *
from src.belief import DiscreteStateFilter
from src.env_utils import GazeActions, MovementActions, action_to_index, index_to_action
from abc import ABC, abstractmethod

from tqdm import tqdm
from gymnasium.spaces import Discrete
import torch

from src.policy_utils import *
from src.mdp import BeliefStateMDP

POLICIES = ["random", "heuristic_sample", "heuristic_greedy", "ffdqn", "convdqn", "rollouts_sample", "rollouts_greedy"]

FAST_POLICIES = ["random", "heuristic_sample", "heuristic_greedy", "ffdqn", "convdqn"]

SLOW_POLICIES = ["rollouts_sample", "rollouts_greedy"]

class Policy(ABC):

    @abstractmethod
    def get_action(self, observation, prev_action):
        pass

    # @abstractmethod
    # def reset(self, observation):
    #     pass

    @staticmethod
    def get(name, env, my_row, my_col):
        assert name in POLICIES, name
        if name == "random":
            return RandomPolicy(env.action_space())
        if name == "heuristic_sample":
            return HeuristicPolicy(my_row, my_col, env.num_rows, env.num_cols, belief_filter=None, mode="sample")
        if name == "heuristic_greedy":
            return HeuristicPolicy(my_row, my_col, env.num_rows, env.num_cols, belief_filter=None, mode="greedy")
        if name == "ffdqn":
            return BeliefDQNPolicy("results/dqn_2024_12_02_00:31:12/policy_final.pth", my_row, my_col, env.num_rows, env.num_cols) # loss=0.8
        if name == "convdqn":
            return BeliefDQNPolicy("results/dqn_2024_12_03_00:56:28/policy_final.pth", my_row, my_col, env.num_rows, env.num_cols) # loss=0.5
        if name == "rollouts_sample":
            return PolicyWithRollouts(my_row, my_col, env.num_rows, env.num_cols, depth=10, num_rollouts=50, bootstrap_mode='sample')
        if name == "rollouts_greedy":
            return PolicyWithRollouts(my_row, my_col, env.num_rows, env.num_cols, depth=10, num_rollouts=50, bootstrap_mode="greedy")
        # if name == "rollouts_sample":
        #     return PolicyWithRollouts(my_row, my_col, env.num_rows, env.num_cols, depth=15, num_rollouts=100, bootstrap_mode='sample')
        # if name == "rollouts_greedy":
        #     return PolicyWithRollouts(my_row, my_col, env.num_rows, env.num_cols, depth=15, num_rollouts=100, bootstrap_mode="greedy")
        
        raise Exception(f"{name} is not a policy!")
        

class PolicyWithRollouts(Policy):
    def __init__(self, my_row, my_col, num_rows, num_cols, depth, num_rollouts, bootstrap_mode):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.depth = depth
        self.num_rollouts = num_rollouts
        self.bootstrap_mode = bootstrap_mode
        self.num_actions = len(MovementActions) * len(GazeActions)
        self.action_space = Discrete(self.num_actions)
        self.mdp = BeliefStateMDP(num_rows, num_cols)
        # the current belief!! always use this
        self.belief_filter = DiscreteStateFilter(my_row, my_col, num_rows, num_cols)

    def __str__(self):
        return f"PolicyWithRollouts({self.depth}, {self.num_rollouts}, {self.bootstrap_mode})"

    # def reset(self, observation):
    #     # raise Exception("dunno how to reset my_row, my_col")
    #     self.belief_filter = DiscreteStateFilter(observation.my_row, observation.my_col, self.num_rows, self.num_cols)

    # def make_rollout_policy(self):
    #     # rollout_policies = [
    #     #     RandomPolicy(self.action_space),
    #     #     HeuristicPolicy(None, None, None, None, belief_filter=self.belief_filter, mode="greedy"),
    #     # ]
    #     # return StochasticMixedPolicy(rollout_policies, [0.01, 0.99])
    #     return HeuristicPolicy(None, None, None, None, belief_filter=self.belief_filter, mode=self.bootstrap_mode)

    def get_action(self, observation, prev_action):
        if prev_action:
            # print("updating belief from")
            # print(self.belief_filter)
            self.belief_filter = update(self.belief_filter, observation, prev_action)
            # print("to")
            # print(self.belief_filter)
        # else:
        #     print(f"no prev action {prev_action}")
        
        action_rewards = {index_to_action(i): [] for i in range(self.num_actions)}

        for _ in tqdm(range(self.num_rollouts)):
        # for _ in tqdm(range(self.num_rollouts)):
            # make_rollout_policy within the for loop because we need to reset the belief to self.belief
            rollout_policy = HeuristicPolicy(observation.my_row, observation.my_col, self.num_rows, self.num_cols, belief_filter=self.belief_filter, mode=self.bootstrap_mode)
            action, discounted_reward = rollout(self.mdp, observation, self.belief_filter, rollout_policy, self.depth)
            action_rewards[action].append(discounted_reward)
        
        """
        pessimistic under uncertainty
        TODO: replace float("-inf") with heuristic estimate
        """
        q_value_estimate = {action: np.mean(rr) if len(rr) > 0 else float("-inf") for action, rr in action_rewards.items()}
        greedy_action = max(q_value_estimate, key=q_value_estimate.get)

        # move_q = {move: 0 for move in [MovementActions.DO_NOTHING, MovementActions.N, MovementActions.E, MovementActions.S, MovementActions.W]}
        # for action, q in q_value_estimate.items():
        #     if q > float("-inf"):
        #         # print(action.move, q)
        #         move_q[action.move] += q
        # for move, q in move_q.items():
        #     print(move, q)
        # print()
        return greedy_action

class StochasticMixedPolicy(Policy):

    def __init__(self, policies, probabilities):
        self.policies = policies
        self.probabilities = probabilities

    def get_action(self, observation, prev_action):
        policy_actions = [(policy, policy.get_action(observation, prev_action)) for policy in self.policies]
        policy, action = random.choices(policy_actions, weights=self.probabilities, k=1)[0]
        # print(f"choosing {action} from rollout policy {policy}")
        return action

    def reset(self, observation):
        for policy in self.policies:
            policy.reset(observation)

class RandomPolicy(Policy):

    def __init__(self, action_space):
        self.action_space = action_space

    # def reset(self, observation):
    #     pass

    def get_action(self, observation=None, prev_action=None):
        index = self.action_space.sample()
        return index_to_action(index)
    
    def __str__(self):
        return "RandomPolicy"


class HeuristicPolicy:
    """
    find which quadrant the opponent is in
    move in one of the directions towards the opponent (break ties randomly)
    gaze in the direction of the opponent (break ties randomly)
    """
    def __init__(self, my_row, my_col, num_rows, num_cols, belief_filter, mode):
        self.mode = mode
        if belief_filter:
            self.belief_filter = belief_filter
        else:
            self.belief_filter = DiscreteStateFilter(my_row, my_col, num_rows, num_cols)
        self._reset_belief_filter = belief_filter

    # def reset(self, observation):
    #     self.belief_filter = self._reset_belief_filter

    def __str__(self):
        return f"SamplingHeuristicPolicy({self.mode})"

    def get_action(self, observation, prev_action):
        # print(self.belief_filter) # for debugging, it is helpful to print the belief filter before the update happens
        if prev_action:
            self.belief_filter = update(self.belief_filter, observation, prev_action)

        # get an action based on the current belief
        my_row, my_col = observation.my_row, observation.my_col
        if self.mode == "greedy":
            target_row, target_col = self.belief_filter.get_center_of_mass()
            # print("center of mass at", target_row, target_col)
        elif self.mode == "sample":
            state = self.belief_filter.sample()
            target_row = state.opp_row
            target_col = state.opp_col
        else:
            raise Exception(f"no such mode {self.mode} for HeuristicPolicy")

        move_actions = []
        if my_row == target_row and my_col == target_col:
            move_actions.append(MovementActions.DO_NOTHING)
        if my_row < target_row:
            move_actions.append(MovementActions.S)
        if my_row > target_row:
            move_actions.append(MovementActions.N)
        if my_col < target_col:
            move_actions.append(MovementActions.E)
        if my_col > target_col:
            move_actions.append(MovementActions.W)

        gaze_actions = []
        if my_row < target_row and my_col < target_col:
            gaze_actions.append(GazeActions.SE)
        elif my_row < target_row and my_col > target_col:
            gaze_actions.append(GazeActions.SW)
        elif my_row > target_row and my_col > target_col:
            gaze_actions.append(GazeActions.NW)
        elif my_row > target_row and my_col < target_col:
            gaze_actions.append(GazeActions.NE)
        
        if not gaze_actions:
            gaze_actions = list(GazeActions)

        move_action = random.choice(move_actions)
        gaze_action = random.choice(gaze_actions)

        # print(f"move: {move_action}, gaze: {gaze_action}")
        return Action(move_action, gaze_action)

class BeliefDQNPolicy(Policy):
    
    def __init__(self, filepath, my_row, my_col, num_rows, num_cols):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
        # print(f"BeliefDQNPolicy device: {self.device}")

        self.filepath = filepath

        n_actions = len(MovementActions) * len(GazeActions)
        n_observations = num_rows * num_cols

        if is_conv_net(filepath):
            self.policy_net = ConvDQN(n_observations, n_actions, None).to(self.device)
        else:
            if is_large_policy(filepath):
                size = "large"
            elif is_medium_policy(filepath):
                size = "medium"
            else:
                size = "small"
            self.policy_net = SimpleDQN(n_observations, n_actions, size=size).to(self.device)
        self.policy_net.load_state_dict(torch.load(self.filepath))
        self.belief_filter = DiscreteStateFilter(my_row, my_col, num_rows, num_cols)

    # def reset(self):
    #     self.belief_filter = DiscreteStateFilter(self.num_rows, self.num_cols)

    def __str__(self):
        return f"BeliefDQNPolicy({self.filepath})"
    
    def get_action(self, observation, prev_action):
        if prev_action:
            self.belief_filter = update(self.belief_filter, observation, prev_action)

        if is_conv_net(self.filepath):
            belief_state = create_dqn_belief_state(observation, self.belief_filter.get_belief(), self.device)
        else:
            belief_state = torch.tensor(self.belief_filter.get_belief_vector(), dtype=torch.float32, device=self.device).unsqueeze(0)
        action_index = get_policy_action(belief_state, self.policy_net).item()
        return index_to_action(action_index)

def is_medium_policy(path):
    return path in ["results/databricks/dqn_2024_12_02_00:05:51/policy_final.pth"]

def is_large_policy(path):
    return path in ["results/dqn_2024_12_02_00:31:12/policy_final.pth"]

def is_conv_net(path):
    return "results/dqn_2024_12_03_00:56:28/" in path
