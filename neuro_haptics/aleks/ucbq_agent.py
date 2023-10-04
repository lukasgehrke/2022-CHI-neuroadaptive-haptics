import numpy as np

class UCBQAgent:
    def __init__(self, params={}):
        # In our case actions == states
        self.num_states = params.get('num_states', 7)
        self.num_actions = params.get('num_actions', 7)
        self.alpha = params.get('alpha', 0.5)  # learning rate
        self.alpha_decay_denumerator = params.get('alpha_decay', 40)
        self.alpha_min = params.get('alpha_min', 0.001)
        self.gamma = params.get('gamma', 0.95)  # discount factor
        # TODO: implement decay. Is it compatible with ucb?
        # TODO: Do we need epsilon greedy?
        # Is there any psychological reason why we can't just switch to the
        # next highest level incrementally?
        self.epsilon = params.get('epsilon', 1)  # epsilon for epsilon-greedy action selection
        self.epsilon_decay_denumerator = params.get('epsilon_decay', 20)
        self.epsilon_min = params.get('epsilon_min', 0.01)        
        # self.epsilon_decay = lambda t: np.log10(t+1)/params.get('epsilon_decay', 20)

        self.Q = np.full((self.num_states, self.num_actions), -6)

        # Initialize N-table for action counts
        # Needs to be `one` to avoid div by zero
        self.N = np.ones((self.num_states, self.num_actions))
        self.t = 0

    def choose_action(self, state):
        self.t += 1

        # Epsilon-greedy action selection
        if np.random.uniform(0, 1) < self.epsilon:
            # Take a random action
            action = np.random.choice(self.num_actions)
        else:
            # Calculate the UCB value for each action
            # TODO: in the original paper there was no `2` but they had a `c`
            ucb_values = self.Q[state] + np.sqrt((2 * np.log(self.t)) / self.N[state])

            # Select action with maximum UCB value
            # Break ties randomly
            idxs_max_values = np.flatnonzero(ucb_values == ucb_values.max())
            action = np.random.choice(idxs_max_values)

        # if self.alpha > self.alpha_min:
        #     alpha_decay = lambda t: np.log10(t+1)/self.alpha_decay_denumerator
        #     self.alpha -= alpha_decay(self.t)

        if self.epsilon > self.epsilon_min:
            epsilon_decay = lambda t: np.log10(t+1)/self.epsilon_decay_denumerator
            self.epsilon -= epsilon_decay(self.t)

        return action

    def learn(self, state, action, reward, next_state):
        # Update N-table for action counts
        self.N[state][action] += 1

        # TODO: double check if this is correct
        self.Q[state][action] = (1 - self.alpha) * self.Q[state][action] \
            + self.alpha * (reward + self.gamma * np.max(self.Q[next_state]))
        # TODO: this is the code from another resource, need to compare if
        # they're equal
        # Update Q-table using Q-learning update rule
        # Qt[i,at] = Qt[i,at] + (Rt - Qt[i,at])/(arm_count[at] + 1)


    def reset(self):
        # Reset Q-table and N-table
        self.Q = np.zeros((self.num_states, self.num_actions))
        self.N = np.ones((self.num_states, self.num_actions))