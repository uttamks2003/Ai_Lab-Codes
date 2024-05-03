
# Q1

class EpsilonGreedyAgent:
    def _init_(self, epsilon):
        self.epsilon = epsilon
        self.Q_values = [0, 0]  # Q-values for two actions (0 and 1)
        self.action_counts = [0, 0]  # Number of times each action has been taken

    def select_action(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(2)  # Explore: randomly choose 0 or 1
        else:
            return np.argmax(self.Q_values)  # Exploit: choose action with highest Q-value

    def update_Q_values(self, action, reward):
        self.action_counts[action] += 1
        step_size = 1 / self.action_counts[action]
        self.Q_values[action] += step_size * (reward - self.Q_values[action])
# Binary bandit A: returns 1 with probability pA, and 0 otherwise
def binaryBanditA():
    pA = 0.5  # Probability of success for bandit A
    return np.random.choice([0, 1], p=[1 - pA, pA])

# Binary bandit B: returns 1 with probability pB, and 0 otherwise
def binaryBanditB():
    pB = 0.7  # Probability of success for bandit B
    return np.random.choice([0, 1], p=[1 - pB, pB])
# Main function to run the epsilon-greedy agent on both bandits
def main():
    epsilon = 0.1
    agent = EpsilonGreedyAgent(epsilon)
    total_rewards_A = 0
    total_rewards_B = 0

    # Lists to store Q-values over time
    Q_values_A_over_time = []
    Q_values_B_over_time = []

    num_steps = 10000
    for t in range(num_steps):
        # Select an action for bandit A and get the reward
        action_A = agent.select_action()
        reward_A = binaryBanditA()
        agent.update_Q_values(action_A, reward_A)
        total_rewards_A += reward_A

        # Select an action for bandit B and get the reward
        action_B = agent.select_action()
        reward_B = binaryBanditB()
        agent.update_Q_values(action_B, reward_B)
        total_rewards_B += reward_B
 # Append current Q-values to the lists
        Q_values_A_over_time.append(agent.Q_values[0])
        Q_values_B_over_time.append(agent.Q_values[1])

    print("Total rewards obtained for Bandit A:", total_rewards_A)
    print("Total rewards obtained for Bandit B:", total_rewards_B)

    # Plotting Q-values over time
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_steps), Q_values_A_over_time, label='Bandit A Q-values')
    plt.plot(range(num_steps), Q_values_B_over_time, label='Bandit B Q-values')
    plt.xlabel('Time Steps')
    plt.ylabel('Q-values')
    plt.title('Q-values over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
    
    # Q2
    
  import numpy as np
import matplotlib.pyplot as plt

def bandit_nonstat(action, m):
    std_dev = 0.01
    v = np.random.normal(0, std_dev, size=10)
    m += v
    value = m[action]
    return value, m

Q = np.zeros(10)
N = np.zeros(10)
R = np.zeros(10000)
epsilon = 0.1
m = np.ones(10)
alpha = 0.7

for i in range(10000):
    if np.random.rand() > epsilon:
        A = np.argmax(Q)
    else:
        temp = np.random.permutation(10)
        A = temp[0]

    RR, m = bandit_nonstat(A, m)
    N[A] += 1
    Q[A] += (RR - Q[A]) * alpha
    if i == 0:
        R[i] = RR
    else:
        R[i] = ((i - 1) * R[i - 1] + RR) / i

plt.plot(range(10000), R, 'r')
plt.xlabel('Time Steps')
plt.ylabel('Average Reward')
plt.title('Average Reward over Time')
plt.grid(True)
plt.show()

# q3

import numpy as np
import matplotlib.pyplot as plt

def non_stat_reward(action, m):
    std_dev = 0.01
    v = np.random.normal(0, std_dev, size=len(m))
    m += v
    reward = m[action]
    return reward, m

Q = np.zeros(10)
N = np.zeros(10)
R = np.zeros(10000)
epsilon = 0.1
m = np.ones(10)
alpha = 0.7

for i in range(10000):
    if np.random.rand() > epsilon:
        A = np.argmax(Q)
    else:
        temp = np.random.permutation(10)
        A = temp[0]

    RR, m = non_stat_reward(A, m)
    N[A] += 1
    Q[A] += (RR - Q[A]) * alpha
    if i == 0:
        R[i] = RR
    else:
        R[i] = ((i - 1) * R[i - 1] + RR) / i

plt.plot(range(10000), R, 'r')
plt.xlabel('Time Steps')
plt.ylabel('Average Reward')
plt.title('Average Reward over Time')
plt.grid(True)
plt.show()  
    
