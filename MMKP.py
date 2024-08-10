import random
import numpy as np

# 입력 데이터
CAPA = [[16, 20], [12, 10]]  #[weight, volume]

WEIGHT = {
    'A': [3, 3],
    'B': [2, 2],
    'C': [4, 4, 4],
    'D': [2],
    'E': [5],
    'F': [3, 3, 3],
    'G': [6, 6]
}

VOLUME = {
    'A': [4, 4],
    'B': [3, 3],
    'C': [2, 2, 2],
    'D': [4],
    'E': [3],
    'F': [5, 5, 5],
    'G': [3, 3]
}

VALUE = {
    'A': [4, 4],
    'B': [5, 5],
    'C': [2, 2, 2],
    'D': [4],
    'E': [3],
    'F': [6, 6, 6],
    'G': [8, 8]
}

# 학습 파라미터
N_EPISODES = 10000
ALPHA = 0.3
GAMMA = 0.9
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.9995

class KnapsackScheduler:
    def __init__(self):
        self.items = list(WEIGHT.keys())
        self.num_items = len(self.items)
        self.num_knapsacks = len(CAPA)
        self.state_size = self.num_items + self.num_knapsacks * 2  # 차원의 개수만큼 num_knapsack에 곱해줌
        self.actions = [(item, knapsack) for item in self.items for knapsack in range(self.num_knapsacks)]
        self.reset()

    def reset(self):
        self.state = [0] * self.num_items + [0] * (self.num_knapsacks * 2)
        self.total_value = 0
        self.selected_items = [[] for _ in range(self.num_knapsacks)]
        return self.state

    def step(self, action):
        item, knapsack = action
        next_state = self.state.copy()
        item_index = self.items.index(item)
        if next_state[item_index] < len(WEIGHT[item]):
            next_state[item_index] += 1
            next_state[self.num_items + knapsack * 2] += WEIGHT[item][next_state[item_index] - 1]
            next_state[self.num_items + knapsack * 2 + 1] += VOLUME[item][next_state[item_index] - 1]
            reward = VALUE[item][next_state[item_index] - 1]
            self.total_value += reward
            self.selected_items[knapsack].append(item)
        else:
            reward = 0

        self.state = next_state
        done = self.is_done()
        return self.state, reward, done

    def is_done(self):
        for i in range(self.num_knapsacks):
            if (self.state[self.num_items + i * 2] > CAPA[i][0] or
                self.state[self.num_items + i * 2 + 1] > CAPA[i][1]):
                return True
        return all(self.state[i] == len(WEIGHT[self.items[i]]) for i in range(self.num_items))

    def is_valid_action(self, action):
        item, knapsack = action
        item_index = self.items.index(item)
        if self.state[item_index] < len(WEIGHT[item]):
            new_weight = self.state[self.num_items + knapsack * 2] + WEIGHT[item][self.state[item_index]]
            new_volume = self.state[self.num_items + knapsack * 2 + 1] + VOLUME[item][self.state[item_index]]
            return new_weight <= CAPA[knapsack][0] and new_volume <= CAPA[knapsack][1]
        return False

class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = {}
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.epsilon = EPSILON_START
        self.epsilon_end = EPSILON_END
        self.epsilon_decay = EPSILON_DECAY

    def get_action(self, state, valid_actions):
        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        state_tuple = tuple(state)
        if state_tuple not in self.q_table:
            self.q_table[state_tuple] = {action: 0 for action in self.action_size}

        return max(self.q_table[state_tuple], key=lambda x: self.q_table[state_tuple].get(x, 0) if x in valid_actions else float('-inf'))

    def update_q_table(self, state, action, reward, next_state, done):
        state_tuple = tuple(state)
        next_state_tuple = tuple(next_state)

        if state_tuple not in self.q_table:
            self.q_table[state_tuple] = {a: 0 for a in self.action_size}

        if next_state_tuple not in self.q_table:
            self.q_table[next_state_tuple] = {a: 0 for a in self.action_size}

        current_q = self.q_table[state_tuple].get(action, 0)
        next_max_q = max(self.q_table[next_state_tuple].values()) if not done else 0
        new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
        self.q_table[state_tuple][action] = new_q

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

def train(n_episodes=N_EPISODES):
    env = KnapsackScheduler()
    agent = QLearningAgent(state_size=env.state_size, action_size=env.actions)

    for episode in range(n_episodes):
        state = env.reset()
        done = False

        while not done:
            valid_actions = [action for action in env.actions if env.is_valid_action(action)]
            if not valid_actions:
                break
            action = agent.get_action(state, valid_actions)
            next_state, reward, done = env.step(action)
            agent.update_q_table(state, action, reward, next_state, done)
            state = next_state

        agent.decay_epsilon()

        if episode % (N_EPISODES//10) == 0:
            print(f"Episode: {episode}, Selected Items: {env.selected_items}, Total Value: {env.total_value}, Knapsack States: {env.state[env.num_items:]}, Epsilon: {agent.epsilon:.4f}")

    return env, agent

def test(env, agent):
    state = env.reset()
    done = False

    while not done:
        valid_actions = [action for action in env.actions if env.is_valid_action(action)]
        if not valid_actions:
            break
        action = agent.get_action(state, valid_actions)
        state, _, done = env.step(action)

    print("Final Selected Items:", env.selected_items)
    print("Total Value:", env.total_value)
    print("Knapsack States (Weight, Volume):", 
          [(env.state[env.num_items + i*2], env.state[env.num_items + i*2 + 1]) for i in range(env.num_knapsacks)])

if __name__ == "__main__":
    env, agent = train()
    test(env, agent)
