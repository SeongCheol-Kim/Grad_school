# load packages
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numpy.random import normal as Gaussian



# define function to generade k probability distributions
class k_armed_data():
    # draw k means, the means of rewards for each of k actions. These resemble the Expectation
    # for each action (random or not), draw from the normal distribution with mean K a value as reward

    def __init__(self, k):
        self.k = k
        self.reward_means = Gaussian(loc=0, scale=1, size=self.k)

    #def __str__(self):
    #    return str(self.reward_means)

    def sample_action(self, action_i):
        return Gaussian(loc=self.reward_means[action_i], scale=1, size=1)[0]


# define function to average rewards incrementally
class q_values():

    def __init__(self, k):
        self.n = np.zeros(k, dtype=int)
        self.q = np.zeros(k)

    #def __str__(self):
    #    return(str(self.q))

    def __repr__(self):
        return(np.array(self.q, dtype=float))

    def update_estimates(self, action_selected, r):
        self.n[action_selected] += 1
        self.q[action_selected] = self.q[action_selected] + (1.0/self.n[action_selected]) * (r - self.q[action_selected])


# define function to run the trial and select actions according to E-greedy
class simple_average_run():

    def __init__(self, k, runs, timeSteps, epsilon):
        self.k = k
        self.runs = runs
        self.timeSteps = timeSteps
        self.epsilon = epsilon

    def select_action(self, q):
        probability = np.random.rand()
        if probability >= self.epsilon:
            return self.greedy_action(q)

        return self.random_action()

    def random_action(self):
        return np.random.choice(self.k)

    def greedy_action(self, q):
        mactions = np.argwhere(q == np.amax(q))
        return(np.random.choice(mactions.flatten().tolist()))

# run trial
if __name__ == "__main__":

    k = 10
    runs = 2000
    timesteps = 1000
    gr_epsilon = 0
    ep_epsilon = 0.1

    greedy_rewards = np.full((runs, timesteps), fill_value=0.)
    epsilon_rewards = np.full((runs, timesteps), fill_value=0.)

    for run in tqdm(range(runs)):

        newData = k_armed_data(k)

        gr_newQvals = q_values(k)
        ep_newQvals = q_values(k)
        gr_currentRun = simple_average_run(k, runs, timesteps, gr_epsilon)
        ep_currentRun = simple_average_run(k, runs, timesteps, ep_epsilon)

        for time in range(timesteps):
            # select action
            gr_currentAction = gr_currentRun.select_action(gr_newQvals.q)
            ep_currentAction = ep_currentRun.select_action(ep_newQvals.q)

            # receive Reward
            gr_currentR = newData.sample_action(gr_currentAction)
            ep_currentR = newData.sample_action(ep_currentAction)

            greedy_rewards[run, time] = gr_currentR
            epsilon_rewards[run, time] = ep_currentR

            # update Q for action with R
            gr_newQvals.update_estimates(gr_currentAction, gr_currentR)
            ep_newQvals.update_estimates(ep_currentAction, ep_currentR)



# plot the results
    gr_mean_rewards = np.average(greedy_rewards, axis=0)
    ep_mean_rewards = np.average(epsilon_rewards, axis=0)
    plt.plot(gr_mean_rewards, label="Ɛ=0.0")
    plt.plot(ep_mean_rewards, label="Ɛ=0.1")

    plt.legend()
    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.show()
