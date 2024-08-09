import random
import numpy as np

# 입력 데이터
WEIGHT_CAPA = 16
VOLUME_CAPA = 20

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
ALPHA = 0.1
GAMMA = 0.9
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.9995

class KnapsackScheduler:
    def __init__(self):
        self.items = list(WEIGHT.keys())
        self.num_items = len(self.items)
        self.state_size = self.num_items
        self.states = [[0] * self.num_items for _ in range(2**self.num_items)]
        self.actions = self.items
        self.reset()

    def reset(self):
        self.state = [0] * self.num_items
        self.total_weight = 0
        self.total_volume = 0
        self.total_value = 0
        self.selected_items = []
        return self.state

    def step(self, action):
        next_state = self.state.copy()
        item_index = self.items.index(action)
        if next_state[item_index] < len(WEIGHT[action]):
            next_state[item_index] += 1
            reward = VALUE[action][next_state[item_index] - 1]
            self.total_weight += WEIGHT[action][next_state[item_index] - 1]
            self.total_volume += VOLUME[action][next_state[item_index] - 1]
            self.total_value += reward
            self.selected_items.append(action)
        else:
            reward = 0

        self.state = next_state
        done = self.total_weight > WEIGHT_CAPA and self.total_volume > VOLUME_CAPA
        return self.state, reward, done

    def is_valid_action(self, action):
        item_index = self.items.index(action)
        if self.state[item_index] < len(WEIGHT[action]):
            new_weight = self.total_weight + WEIGHT[action][self.state[item_index]]
            new_volume = self.total_volume + VOLUME[action][self.state[item_index]]
            return new_weight <= WEIGHT_CAPA and new_volume <= VOLUME_CAPA
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

        if tuple(state) not in self.q_table:
            self.q_table[tuple(state)] = {action: 0 for action in WEIGHT.keys()}

        return max(self.q_table[tuple(state)], key=self.q_table[tuple(state)].get)

    def update_q_table(self, state, action, reward, next_state, done):
        if tuple(state) not in self.q_table:
            self.q_table[tuple(state)] = {action: 0 for action in WEIGHT.keys()}

        if tuple(next_state) not in self.q_table:
            self.q_table[tuple(next_state)] = {action: 0 for action in WEIGHT.keys()}

        current_q = self.q_table[tuple(state)][action]
        next_max_q = max(self.q_table[tuple(next_state)].values()) if not done else 0
        new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
        self.q_table[tuple(state)][action] = new_q

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

def train(n_episodes=N_EPISODES):
    env = KnapsackScheduler()
    agent = QLearningAgent(state_size=env.state_size, action_size=len(env.actions))

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
            print(f"Episode: {episode}, Selected Items: {env.selected_items}, Total Value: {env.total_value}, Total Weight: {env.total_weight}, Total Volume: {env.total_volume}, Epsilon: {agent.epsilon:.4f}")

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
    print("Total Weight:", env.total_weight)
    print("Total Volume:", env.total_volume)


if __name__ == "__main__":
    env, agent = train()
    test(env, agent)
