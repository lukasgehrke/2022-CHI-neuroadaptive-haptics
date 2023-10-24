import modified_pendulum_processor
import noise_estimator
import numpy as np
import collections

num_actions = 7

class ModifiedPendulumProcessorNoiseless(modified_pendulum_processor.ModifiedPendulumProcessor):
    # This one is surrogate by default
    def __init__(self):
        super().__init__(weight=0.2, surrogate=True, noise_type="anti_iden", epsilon=1e-6)

    def process_step(self, observation, reward, done, info, action):
        state = observation
        self.action = action

        self.r_sum += reward
        self.r_counter += 1

        reward = int(np.ceil(reward))
        # Difference with the original one - this one doesn't
        # add noise to the reward
        # reward = self.noisy_reward(reward)
        self.collect(state, self.action, reward)
        reward = self.process_reward(reward)

        return observation, reward, done, info