from ucbq_environment import ModifiedRandomEnvironment

class ModifiedRandomEnvironment(ModifiedRandomEnvironment):
    def __init__(self, correct_action = 6):
        # The total number of feedback levels
        self.num_states = 1
        # The last feedback level sent
        self.current_state = 0
        # The "right" level of feedback
        self.correct_action = correct_action        
        
        self.same_action = None
        self.t = 0
        self.consecutive_limit = 15
        self.consecutive_count = 0

    def step(self, action):
        self.t += 1

        reward = self.get_participant_answer(action)
        # TODO: delete
        # Our case action == state, but migth consdier this separation in the future
        next_state = 0
        self.current_state = next_state
        
        done = False
        if self.t > 35:
            if action == self.same_action:
                self.consecutive_count += 1
                if self.consecutive_count >= self.consecutive_limit:
                    # tqdm.write(f"Selected action {action} - breaking loop at iteration {t} - consecutive count: {consecutive_count}")
                    done = True
            else:
                self.same_action = action
                self.consecutive_count = 1
        
        return reward, next_state, done