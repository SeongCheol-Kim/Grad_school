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
    ep1_epsilon = 0.01
    ep2_epsilon = 0.1

    greedy_rewards = np.full((runs, timesteps), fill_value=0.)
    epsilon1_rewards = np.full((runs, timesteps), fill_value=0.)
    epsilon2_rewards = np.full((runs, timesteps), fill_value=0.)

    for run in tqdm(range(runs)):

        newData = k_armed_data(k)

        gr_newQvals = q_values(k)
        ep1_newQvals = q_values(k)
        ep2_newQvals = q_values(k)
        gr_currentRun = simple_average_run(k, runs, timesteps, gr_epsilon)
        ep1_currentRun = simple_average_run(k, runs, timesteps, ep1_epsilon)
        ep2_currentRun = simple_average_run(k, runs, timesteps, ep2_epsilon)

        for time in range(timesteps):
            # select action
            gr_currentAction = gr_currentRun.select_action(gr_newQvals.q)
            ep1_currentAction = ep1_currentRun.select_action(ep1_newQvals.q)
            ep2_currentAction = ep2_currentRun.select_action(ep2_newQvals.q)

            # receive Reward
            gr_currentR = newData.sample_action(gr_currentAction)
            ep1_currentR = newData.sample_action(ep1_currentAction)
            ep2_currentR = newData.sample_action(ep2_currentAction)

            greedy_rewards[run, time] = gr_currentR
            epsilon1_rewards[run, time] = ep1_currentR
            epsilon2_rewards[run, time] = ep2_currentR

            # update Q for action with R
            gr_newQvals.update_estimates(gr_currentAction, gr_currentR)
            ep1_newQvals.update_estimates(ep1_currentAction, ep1_currentR)
            ep2_newQvals.update_estimates(ep2_currentAction, ep2_currentR)



# plot the results
    gr_mean_rewards = np.average(greedy_rewards, axis=0)
    ep1_mean_rewards = np.average(epsilon1_rewards, axis=0)
    ep2_mean_rewards = np.average(epsilon2_rewards, axis=0)
    plt.plot(gr_mean_rewards, label="Ɛ=0.0")
    plt.plot(ep1_mean_rewards, label="Ɛ=0.01")
    plt.plot(ep2_mean_rewards, label="Ɛ=0.1")

    plt.legend()
    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.show()
