import numpy as np

class UCBQAgent:
    def __init__(self, num_states=7, num_actions=7, alpha=0.5, gamma=0.95, epsilon=1.0):
        # In our case actions == states
        self.num_states = num_states # num feedback levels
        self.num_actions = num_actions # num feedback levels
        self.alpha = alpha  # learning rate
        self.alpha_decay = lambda t: np.log10((t+1)/40)
        self.alpha_min = 0.001
        self.gamma = gamma  # discount factor
        # TODO: implement decay. Is it compatible with ucb?
        # TODO: Do we need epsilon greedy?
        # Is there any psychological reason why we can't just switch to the
        # next highest level incrementally?
        self.epsilon = epsilon  # epsilon for epsilon-greedy action selection
        # self.epsilon_decay = 0.8
        self.epsilon_decay = lambda t: np.log10((t+1)/20)        
        self.epsilon_min = 0.01


        # Initialize Q-table with zeros
        # TODO:
        # Should we initialize this to `-1` instead of zeroes initially?
        self.Q = np.zeros((self.num_states, self.num_actions))

        # Initialize N-table for action counts
        # Needs to be `one` to avoid div by zero
        self.N = np.ones((self.num_states, self.num_actions))

    def choose_action(self, state):
        # Epsilon-greedy action selection
        if np.random.uniform(0, 1) < self.epsilon:
            # Take a random action
            action = np.random.choice(self.num_actions)
        else:
            # Calculate the UCB value for each action
            t = sum(self.N[state])
            # TODO: in the original paper there was no `2` but they had a `c`
            ucb_values = self.Q[state] + np.sqrt((2 * np.log(t)) / self.N[state])

            # Select action with maximum UCB value
            action = np.argmax(ucb_values)

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