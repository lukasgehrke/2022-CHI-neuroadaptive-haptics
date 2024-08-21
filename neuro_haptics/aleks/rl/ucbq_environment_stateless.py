from .ucbq_environment import ModifiedRandomEnvironment

class ModifiedRandomEnvironment(ModifiedRandomEnvironment):
    def __init__(self, params={}):
        super().__init__(params=params)        
        # The total number of feedback levels
        self.num_states = 1
        # The last feedback level sent
        self.current_state = 0
        
        self.same_action = None
        self.t = 0
        self.convergence_count_start = params.get('convergence_count_start', 35)
        self.convergence_consecutive_limit = params.get('convergence_consecutive_limit', 15)
        self.consecutive_count = 0


    def step(self, action):
        self.t += 1

        reward = self.send_feedback_to_participant_and_get_participant_answer(action)
        # TODO: delete
        # Our case action == state, but migth consider this separation in the future
        next_state = 0
        self.current_state = next_state
        
        done = False
        if self.t > self.convergence_count_start:
            if action == self.same_action:
                self.consecutive_count += 1
                if self.consecutive_count >= self.convergence_consecutive_limit:
                    # tqdm.write(f"Selected action {action} - breaking loop at iteration {t} - consecutive count: {consecutive_count}")
                    done = True
            else:
                self.same_action = action
                self.consecutive_count = 1
        
        return reward, next_state, done