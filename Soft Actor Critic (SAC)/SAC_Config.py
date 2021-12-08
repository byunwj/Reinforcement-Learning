class Config(object):
    def __init__(self):
        # parameters
        self.BATCH_SIZE = 128
        self.EPISODES = 400
        self.ACTOR_LR = 0.001                 # actor learning rate
        self.CRITIC_LR = 0.002                # critic learning rate
        self.ALPHA_LR = 0.001
        self.GAMMA = 0.9                      # discount rate for the bellman equation
        self.TAU = 0.01                       # soft target update
        self.COUNTER = 0                      # replay buffer counter
        self.MEMORY_CAPACITY = 10000          # maximum experience queue size
        self.RENDER = False                   # render toggle
