import numpy as np


class Badit():
    # 定义一个摇臂机 包含构造函数和摇臂函数
    def __init__(self, k, probability=None):
        self.k = k
        self.probability = probability if probability is not None else np.random.rand(
            k)

    def step(self, action):
        reward = 1 if np.random.rand() < self.probability[action] else 0
        return reward


class Agent():
    # 定义进行学习的智能体，构造函数包括k个摇臂，贪婪度，以及每个摇臂的价值
    def __init__(self, badit, epsilon=0.1):
        self.K = np.zeros(badit.k)
        self.epsilon = epsilon
        self.Q = np.zeros(badit.k)
        self.badit = badit

    def reset(self):
        self.Q = np.zeros(self.badit.k)
        self.K = np.zeros(self.badit.k)

    def choose_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.badit.k)
        else:
            # 如果有多个最大值则从多个最大值中随机选择一个
            return np.random.choice(np.where(self.Q == np.max(self.Q))[0])

    def learn(self, action, reward):
        # 新的Q_k = (Q_{k-1}*(k-1) + reward)/k， 进行一个化简就可以变成
        # Q_k = Q_{k-1} + (reward - Q_{k-1})/k
        self.K[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.K[action]


badit = Badit(6)
agent = Agent(badit)
print('badit settings:' + str(np.around(badit.probability, 4)))
for i in range(100000):
    action = agent.choose_action()
    reward = badit.step(action)
    agent.learn(action, reward)
    if i % 1000 == 0:
        print('step: {:5d},      choose: {},      reward: {},      Q: {}'.format(
            i, action+1, reward, str(np.around(agent.Q, 4))))

# 进行一个10w次的跑，输出结果发现Q逐渐向摇臂机的概率靠拢
# 笔记：如果已经足够近似了就不再需要探索了，所以可以让epsilon逐渐减小，比如epsilon = 1/sqrt(i)