import pygame
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import math
from statistics import mean
import matplotlib.pyplot as plt
from queue import Queue

# ======================== CONFIGURATION ========================
GRID_SIZE = 15
CELL_SIZE = 25
WINDOW_SIZE = GRID_SIZE * CELL_SIZE
NUM_AGENTS = 5
EPISODES = 1000
MAX_STEPS = 1000
BATCH_SIZE = 64
MEMORY_CAPACITY = 10000
LEARNING_RATE = 1e-4
GAMMA = 0.95
EPSILON_START = 0.9
EPSILON_END = 0.05
EPSILON_DECAY = 0.999
TARGET_UPDATE_FREQ = 50

# ======================== COLOR DEFINITIONS ========================
WHITE = (1.0, 1.0, 1.0)
BLACK = (0.0, 0.0, 0.0)
GRAY = (0.588, 0.588, 0.588)
BROWN = (0.396, 0.263, 0.129)
ORANGE = (1.0, 0.647, 0.0)
GREEN = (0.0, 1.0, 0.0)
AGENT_COLORS = [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0), (1.0, 1.0, 1.0)]

# ======================== WAREHOUSE ENVIRONMENT ========================
class WarehouseEnv:
    def __init__(self):
        self.size = GRID_SIZE
        self.grid_map = self.generate_map()
        self.pickup_points = self.generate_pickup_points()
        self.delivery_points = self.generate_delivery_points()
        self.agent_starts = self.generate_start_positions()
        self.reset()
        
    def generate_map(self):
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        for col in range(2, GRID_SIZE-2, 3):
            for row in range(2, GRID_SIZE-2):
                if row % 3 != 0:
                    grid[row, col] = 1
        for row in [3, 6]:
            for col in range(1, GRID_SIZE-1):
                if random.random() < 0.4:
                    grid[row, col] = 1
        for _ in range(10):
            row, col = random.randint(1, GRID_SIZE-2), random.randint(1, GRID_SIZE-2)
            if grid[row, col] == 0 and (row, col) not in self.generate_pickup_points() and (row, col) not in self.generate_delivery_points():
                grid[row, col] = 1
        for i in range(NUM_AGENTS):
            start = self.generate_start_positions()[i]
            pickup = self.generate_pickup_points()[i]
            delivery = self.generate_delivery_points()[i]
            if not (self.has_path(start, pickup, grid) and self.has_path(pickup, delivery, grid)):
                grid = self.clear_obstacles(grid, start, pickup, delivery)
        return grid
    
    def has_path(self, start, end, grid):
        visited = set()
        queue = Queue()
        queue.put(start)
        visited.add(start)
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        while not queue.empty():
            x, y = queue.get()
            if (x, y) == end:
                return True
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and grid[nx, ny] != 1 and (nx, ny) not in visited:
                    queue.put((nx, ny))
                    visited.add((nx, ny))
        return False
    
    def clear_obstacles(self, grid, start, pickup, delivery):
        for row in range(min(start[0], pickup[0]), max(start[0], pickup[0]) + 1):
            grid[row, start[1]] = 0
        for col in range(min(start[1], pickup[1]), max(start[1], pickup[1]) + 1):
            grid[pickup[0], col] = 0
        for row in range(min(pickup[0], delivery[0]), max(pickup[0], delivery[0]) + 1):
            grid[row, delivery[1]] = 0
        for col in range(min(pickup[1], delivery[1]), max(pickup[1], delivery[1]) + 1):
            grid[delivery[0], col] = 0
        return grid
    
    def generate_pickup_points(self):
        y_positions = np.linspace(1, GRID_SIZE - 2, NUM_AGENTS, dtype=int)
        return [(GRID_SIZE-2, y) for y in y_positions]
    
    def generate_delivery_points(self):
        y_positions = np.linspace(1, GRID_SIZE - 2, NUM_AGENTS, dtype=int)
        return [(1, y) for y in y_positions]
    
    def generate_start_positions(self):
        return [(GRID_SIZE-2, 1 + 3*i) for i in range(NUM_AGENTS)]
    
    def reset(self):
        self.agent_positions = list(self.agent_starts)
        self.has_package = [False] * NUM_AGENTS
        self.completed = [False] * NUM_AGENTS
        self.steps = 0
        return self.get_states()
    
    def get_states(self):
        states = []
        for i in range(NUM_AGENTS):
            x, y = self.agent_positions[i]
            px, py = self.pickup_points[i]
            dx, dy = self.delivery_points[i]
            local_map = np.zeros((5, 5))
            for i2 in range(-2, 3):
                for j2 in range(-2, 3):
                    nx, ny = x + i2, y + j2
                    if 0 <= nx < self.size and 0 <= ny < self.size:
                        local_map[i2 + 2, j2 + 2] = self.grid_map[nx, ny]
                    else:
                        local_map[i2 + 2, j2 + 2] = 1
            state = np.concatenate([
                [x/GRID_SIZE, y/GRID_SIZE],
                [px/GRID_SIZE, py/GRID_SIZE],
                [dx/GRID_SIZE, dy/GRID_SIZE],
                [1 if self.has_package[i] else 0],
                local_map.flatten()
            ])
            states.append(state)
        return states
    
    def step(self, actions):
        rewards = [0] * NUM_AGENTS
        next_states = []
        self.steps += 1
        prev_distances = []
        for i in range(NUM_AGENTS):
            x, y = self.agent_positions[i]
            px, py = self.pickup_points[i]
            dx, dy = self.delivery_points[i]
            target = (px, py) if not self.has_package[i] else (dx, dy)
            prev_distances.append(math.sqrt((x - target[0])**2 + (y - target[1])**2))
        new_positions = []
        target_positions = set()
        occupied = set(self.agent_positions)
        for i, action in enumerate(actions):
            if self.completed[i]:
                new_positions.append(self.agent_positions[i])
                target_positions.add(self.agent_positions[i])
                continue
            x, y = self.agent_positions[i]
            move_x, move_y = [(0,0), (-1,0), (1,0), (0,-1), (0,1)][action]
            nx, ny = x + move_x, y + move_y
            if (nx, ny) == self.delivery_points[i]:
                new_positions.append((nx, ny))
                target_positions.add((nx, ny))
                continue
            if (0 <= nx < self.size and 0 <= ny < self.size and 
                self.grid_map[nx, ny] != 1 and 
                (nx, ny) not in target_positions and 
                (nx, ny) not in occupied - {(x, y)}):
                new_positions.append((nx, ny))
                target_positions.add((nx, ny))
            else:
                new_positions.append((x, y))
                rewards[i] -= 0.1
        self.agent_positions = new_positions
        for i in range(NUM_AGENTS):
            if self.completed[i]:
                next_states.append(self.get_states()[i])
                continue
            x, y = self.agent_positions[i]
            px, py = self.pickup_points[i]
            dx, dy = self.delivery_points[i]
            target = (px, py) if not self.has_package[i] else (dx, dy)
            curr_dist = math.sqrt((x - target[0])**2 + (y - target[1])**2)
            prev_dist = prev_distances[i]
            rewards[i] += (prev_dist - curr_dist) * 0.5
            if not self.has_package[i] and (x, y) == (px, py):
                rewards[i] += 20
                self.has_package[i] = True
            elif self.has_package[i] and (x, y) == (dx, dy):
                rewards[i] += 50
                self.completed[i] = True
                self.has_package[i] = False
            rewards[i] -= 0.01
            next_states.append(self.get_states()[i])
        done = all(self.completed) or (self.steps >= MAX_STEPS)
        if all(self.completed):
            completion_bonus = (MAX_STEPS - self.steps) * 0.1
            for i in range(NUM_AGENTS):
                rewards[i] += completion_bonus
        return next_states, rewards, done

# ======================== DQN MODEL ========================
class DQN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )
    def forward(self, x):
        return self.network(x)

# ======================== MULTI-AGENT SYSTEM ========================
class MultiAgentSystem:
    def __init__(self, state_size):
        self.agents = [DQNAgent(i, state_size) for i in range(NUM_AGENTS)]
        self.steps = 0
    
    def act(self, states):
        return [agent.act(states[i]) for i, agent in enumerate(self.agents)]
    
    def remember(self, states, actions, rewards, next_states, done):
        for i, agent in enumerate(self.agents):
            agent.remember(states[i], actions[i], rewards[i], next_states[i], done)
    
    def replay(self):
        for agent in self.agents:
            agent.replay()
        self.steps += 1
        if self.steps % TARGET_UPDATE_FREQ == 0:
            for agent in self.agents:
                agent.update_target_network()
        for agent in self.agents:
            agent.update_epsilon()

class DQNAgent:
    def __init__(self, agent_id, state_size):
        self.model = DQN(state_size)
        self.target_model = DQN(state_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.memory = deque(maxlen=MEMORY_CAPACITY)
        self.agent_id = agent_id
        self.epsilon = EPSILON_START
    
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 4)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        minibatch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        current_q = self.model(states).gather(1, actions)
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + GAMMA * next_q * (1 - dones)
        loss = nn.functional.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def update_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

# ======================== VISUALIZATION ========================
def draw_env(screen, env):
    screen.fill(tuple(int(c * 255) for c in WHITE))
    for x in range(0, WINDOW_SIZE, CELL_SIZE):
        pygame.draw.line(screen, tuple(int(c * 255) for c in GRAY), (x, 0), (x, WINDOW_SIZE))
    for y in range(0, WINDOW_SIZE, CELL_SIZE):
        pygame.draw.line(screen, tuple(int(c * 255) for c in GRAY), (0, y), (WINDOW_SIZE, y))
    for i in range(env.size):
        for j in range(env.size):
            if env.grid_map[i, j] == 1:
                rect = pygame.Rect(j*CELL_SIZE, i*CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, tuple(int(c * 255) for c in BROWN), rect)
    for i, (px, py) in enumerate(env.pickup_points):
        rect = pygame.Rect(py*CELL_SIZE, px*CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, tuple(int(c * 255) for c in ORANGE), rect)
        font = pygame.font.SysFont(None, 20)
        text = font.render(f"P{i}", True, tuple(int(c * 255) for c in BLACK))
        screen.blit(text, (py*CELL_SIZE+5, px*CELL_SIZE+5))
    for i, (dx, dy) in enumerate(env.delivery_points):
        rect = pygame.Rect(dy*CELL_SIZE, dx*CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, tuple(int(c * 255) for c in GREEN), rect)
        font = pygame.font.SysFont(None, 20)
        text = font.render(f"D{i}", True, tuple(int(c * 255) for c in BLACK))
        screen.blit(text, (dy*CELL_SIZE+5, dx*CELL_SIZE+5))
    for i, (x, y) in enumerate(env.agent_positions):
        rect = pygame.Rect(y*CELL_SIZE+5, x*CELL_SIZE+5, CELL_SIZE-10, CELL_SIZE-10)
        color = GRAY if env.completed[i] else (BLACK if env.has_package[i] else AGENT_COLORS[i])
        pygame.draw.rect(screen, tuple(int(c * 255) for c in color), rect)
        font = pygame.font.SysFont(None, 20)
        text = font.render(str(i), True, tuple(int(c * 255) for c in (WHITE if color == BLACK else BLACK)))
        screen.blit(text, (y*CELL_SIZE+10, x*CELL_SIZE+10))
    pygame.display.flip()

# ======================== PLOTTING FUNCTION ========================
def plot_metrics(agent_rewards, avg_steps, avg_losses):
    plt.figure(figsize=(10, 6))
    for i in range(NUM_AGENTS):
        plt.plot(range(1, len(agent_rewards[i]) + 1), agent_rewards[i], label=f'Agent {i}', color=AGENT_COLORS[i])
    plt.title('Per-Agent Rewards Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig('rewards.png')
    plt.close()
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(avg_steps) + 1), avg_steps, 'b-', label='Average Steps')
    plt.title('Average Steps Per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.legend()
    plt.grid(True)
    plt.savefig('avg_steps.png')
    plt.close()
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(avg_losses) + 1), avg_losses, 'r-', label='Average Loss')
    plt.title('Average Loss Per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('avg_loss.png')
    plt.close()

# ======================== MAIN TRAINING LOOP ========================
def train():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Multi-Agent Warehouse with Standard DQN")
    clock = pygame.time.Clock()
    env = WarehouseEnv()
    initial_state = env.get_states()[0]
    state_size = len(initial_state)
    multi_agent = MultiAgentSystem(state_size)
    agent_rewards = [[] for _ in range(NUM_AGENTS)]
    avg_steps = []
    avg_losses = []
    for ep in range(EPISODES):
        states = env.reset()
        total_rewards = [0] * NUM_AGENTS
        successes = [0] * NUM_AGENTS
        losses = [[] for _ in range(NUM_AGENTS)]
        render = True
        for t in range(MAX_STEPS):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    plot_metrics(agent_rewards, avg_steps, avg_losses)
                    return
            actions = multi_agent.act(states)
            next_states, rewards, done = env.step(actions)
            for i, agent in enumerate(multi_agent.agents):
                if len(agent.memory) >= BATCH_SIZE:
                    minibatch = random.sample(agent.memory, BATCH_SIZE)
                    states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = zip(*minibatch)
                    states_batch = torch.tensor(np.array(states_batch), dtype=torch.float32)
                    actions_batch = torch.tensor(actions_batch, dtype=torch.long).unsqueeze(1)
                    rewards_batch = torch.tensor(rewards_batch, dtype=torch.float32).unsqueeze(1)
                    next_states_batch = torch.tensor(np.array(next_states_batch), dtype=torch.float32)
                    dones_batch = torch.tensor(dones_batch, dtype=torch.float32).unsqueeze(1)
                    current_q = agent.model(states_batch).gather(1, actions_batch)
                    with torch.no_grad():
                        next_q = agent.target_model(next_states_batch).max(1)[0].unsqueeze(1)
                        target_q = rewards_batch + GAMMA * next_q * (1 - dones_batch)
                    loss = nn.functional.mse_loss(current_q, target_q)
                    losses[i].append(loss.item())
            multi_agent.remember(states, actions, rewards, next_states, done)
            multi_agent.replay()
            for i in range(NUM_AGENTS):
                total_rewards[i] += rewards[i]
                if env.completed[i]:
                    successes[i] = 1
            states = next_states
            if render:
                draw_env(screen, env)
                clock.tick(60)
            if done:
                break
        for i in range(NUM_AGENTS):
            agent_rewards[i].append(total_rewards[i])
        avg_steps.append(env.steps)
        avg_losses.append(mean([mean(l) if l else 0 for l in losses]))
        print(f"Episode {ep+1}/{EPISODES}, Total Rewards: {total_rewards}, "
              f"Steps: {env.steps}, Epsilon: {multi_agent.agents[0].epsilon:.3f}, "
              f"Successes: {successes}, Avg Loss: {[mean(l) if l else 0 for l in losses]}")
    pygame.quit()
    plot_metrics(agent_rewards, avg_steps, avg_losses)

if __name__ == "__main__":
    train()