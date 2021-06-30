class ExperienceBuffer:
    def __init__(self, meta=None, observations=None, actions=None, rewards=None, dones=None):
        self.meta = meta
        self.result = 0
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []

        if observations is not None:
            self.observations = observations

        if actions is not None:
            self.actions = actions

        if rewards is not None:
            self.rewards = rewards

        if dones is not None:
            self.dones = dones

    def size(self):
        return len(self.dones)

    def add_step(self, observation, action, reward, done):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def generate_rollouts(self, batch_size):
        for i in range(0, len(self.observations), batch_size):
            yield ExperienceBuffer(self.observations[i:i+batch_size],
                                   self.actions[i:i+batch_size],
                                   self.rewards[i:i+batch_size],
                                   self.dones[i:i+batch_size])


