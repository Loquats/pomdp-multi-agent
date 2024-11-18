def template_policy(observation, action_space, agent):
    pass

def heuristic_policy(observation, action_space, agent):
    """
    find which quadrant the opponent is in
    move in one of the directions towards the opponent (break ties randomly)
    gaze in the direction of the opponent (break ties randomly)
    """
    pass

def random_policy(action_space):
    return action_space.sample()