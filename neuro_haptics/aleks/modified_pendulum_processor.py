import noise_estimator
import numpy as np
import collections

import random
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class ModifiedPendulumProcessor(noise_estimator.PendulumProcessor):
    """
    Learning from perturbed rewards -- Pendulum
    step 1 - Estimate the confusion matrices (17 x 17)
    step 2 - Calculate the surrogate rewards
    """
    def __init__(self, weight=0.2, surrogate=False, noise_type="anti_iden", epsilon=1e-6, surrogate_c_interval=10, 
                 num_unique_rewards=None,
                 diag=0.5):
        self.r_sets = {}
        self.weight = weight
        self.surrogate = surrogate

        self.M = num_unique_rewards
        # self.cmat, _ = noise_estimator.initialize_cmat(noise_type, self.M, self.weight)
        self.cmat = self.initialize_cmat(diag=diag)
        # assert (is_invertible(self.cmat))
        # self.cummat = np.cumsum(self.cmat, axis=1)
        self.mmat = np.expand_dims(np.asarray(range(0, -1 * self.M, -1)), axis=1)

        self.r_sum = 0
        self.r_counter = 0

        self.counter = 0
        self.C = np.identity(self.M)
        self.epsilon = epsilon

        if self.weight > 0.5:
            self.reverse = True
        else: self.reverse = False
        self.valid = False

        self.surrogate_c_interval = surrogate_c_interval

    def initialize_cmat(self, diag=0.5):
        confusion_matrix = np.zeros((self.M, self.M))
        diag_1 = (1-diag) * (0.8/2)
        diag_2 = (1-diag) * (0.2/2)
        np.fill_diagonal(confusion_matrix, diag)
        np.fill_diagonal(confusion_matrix[:, 1:], diag_1)
        np.fill_diagonal(confusion_matrix[1:, :], diag_1)
        np.fill_diagonal(confusion_matrix[:, 2:], diag_2)
        np.fill_diagonal(confusion_matrix[2:, :], diag_2)

        # Normalize the rows
        row_sums = confusion_matrix.sum(axis=1)
        confusion_matrix = confusion_matrix / row_sums[:, np.newaxis]
        confusion_matrix = np.around(confusion_matrix, decimals=4)

        return confusion_matrix

    # def noisy_reward(self, reward):
    #     prob_list = list(self.cummat[abs(reward), :])
    #     n = np.random.random()
    #     prob_list.append(n)
    #     j = sorted(prob_list).index(n)
    #     reward = -1 * j

    #     return reward

    def noisy_reward(self, reward):
        prob_list = list(self.cmat[abs(reward), :])
        # Use random.choices to pick an index based on probabilities
        chosen_index = random.choices(range(len(prob_list)), weights=prob_list, k=1)[0]
        reward = -1 * chosen_index

        return reward

    def process_reward(self, reward):
        if not self.surrogate:
            return reward

        self.estimate_C()

        if self.valid:
            # TODO: They didn't use the learning rate!
            return self.phi[int(-reward), 0]
        else: return reward

    def estimate_C(self):
        # if self.counter >= 1000 and self.counter % 100 == 0:
        if self.counter >= self.surrogate_c_interval and self.counter % self.surrogate_c_interval == 0:        
            self.C = np.zeros((self.M, self.M))
            # self.C = np.identity(self.M)
            self.count = [0] * self.M

            for k in self.r_sets.keys():
                freq_count = collections.Counter(self.r_sets[k])

                if self.reverse:
                    truth, count = freq_count.most_common()[-1]
                else:
                    truth, count = freq_count.most_common()[0]

                self.count[int(-truth)] += len(self.r_sets[k])
            # print (self.count)

            for k in self.r_sets.keys():
                freq_count = collections.Counter(self.r_sets[k])
                list_freq = freq_count.most_common()
                if self.reverse:
                    list_freq = sorted(list_freq, reverse=True)
                truth, count = list_freq[0]
                # if self.first_time[int(-truth)]:
                #     self.C[int(-truth), int(-truth)] = 0
                    # self.first_time[int(-truth)] = False
                # print (prob_correct)

                for pred, count in list_freq:
                    self.C[int(-truth), int(-pred)] += float(count) / self.count[int(-truth)]

            # diag = np.diag(self.C)
            # anti_diag = np.diag(np.fliplr(self.C))
            # log_string("diag: " + np.array2string(diag, formatter={'float_kind':lambda x: "%.5f" % x}))
            # log_string("anti_diag:" + np.array2string(anti_diag, formatter={'float_kind':lambda x: "%.5f" % x}))
            # log_string("sum: " + np.array2string(np.sum(self.C, axis=1), formatter={'float_kind':lambda x: "%.2f" % x}))

            # self.C = self.C * np.identity(self.M)
            # self.C = init_norm.sum(axis=1, keepdims=1)

            if noise_estimator.is_invertible(self.C):
                # they're pre-multiplying the rewards with the matrix here
                self.phi = np.linalg.inv(self.C).dot(self.mmat)
                self.valid = True
            else: self.valid = False

    def collect(self, state, action, reward):
        if (state, action) in self.r_sets:
            self.r_sets[(state, action)].append(reward)
        else:
            self.r_sets[(state, action)] = [reward]
        self.counter += 1

    # def process_action(self, action):
    #     # print ("action before:", action)
    #     n_bins = 20
    #     action_bins = pandas.cut([-1.0, 1.0], bins=n_bins, retbins=True)[1][1:-1]
    #     self.action = build_state([to_bin(action, action_bins)])
    #     # print ("action after:", self.action)

    #     return action

    def process_step(self, observation, reward, done, info, action):
        state = observation
        self.action = action

        self.r_sum += reward
        self.r_counter += 1

        # if self.r_counter == 200:
        #     # log_string(str(self.r_sum / float(self.r_counter)))
        #     self.r_counter = 0
        #     self.r_sum = 0

        reward = int(np.ceil(reward))
        reward = self.noisy_reward(reward)
        self.collect(state, self.action, reward)
        reward = self.process_reward(reward)

        return observation, reward, done, info
    
    def print(self):
        # print('Original noise/confusion matrix:')
        # ConfusionMatrixDisplay(self.cmat).plot()
        # plt.show()        
        print('Estimated confusion matrix:')
        estimated_C = np.around(self.C, decimals=4)
        ConfusionMatrixDisplay(estimated_C).plot()
        plt.show()
        
        r_sets_sorted = sorted(self.r_sets.items(), key=lambda item: item[0][1])

        print('Reward sets:')
        for key, value in r_sets_sorted:
            print(f"{key}: {value}")
        
        print('Reward set counts:')
        key_counts = {key: len(np.array(value)) for key, value in r_sets_sorted}
        for key, count in key_counts.items():
            print(f"Key {key}: {count} items")
