# rl_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import networkx as nx

# --- DQN Agent Model ---
class DQNAgent(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNAgent, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# --- Base Environment Class ---
class Env:
    def __init__(self, graph, start_node, end_node):
        self.graph = graph
        self.start_node = start_node
        self.end_node = end_node
        self.current_node = start_node
        self.max_neighbors = 5

    def get_state(self):
        neighbors = list(self.graph.neighbors(self.current_node))
        features = []
        for neighbor in neighbors[:self.max_neighbors]:
            edge_data = self.graph.get_edge_data(self.current_node, neighbor)
            
            if isinstance(edge_data, dict) and 0 in edge_data:
                edge_data = edge_data[0]
            
            aqi = edge_data.get('aqi', 3)
            greenery = edge_data.get('greenery_score', 0)
            distance = edge_data.get('length', 1)
            features.extend([aqi, greenery, distance])

        while len(features) < self.max_neighbors * 3:
            features.extend([0, 0, 0])

        return np.array(features), neighbors

    def step(self, action):
        neighbors = list(self.graph.neighbors(self.current_node))

        if len(neighbors) == 0:
            return self.get_state(), -100, True

        if action >= len(neighbors):
            chosen = neighbors[0] 
        else:
            chosen = neighbors[action]

        edge_data = self.graph.get_edge_data(self.current_node, chosen)
        if isinstance(edge_data, dict) and 0 in edge_data:
            edge_data = edge_data[0]

        aqi = edge_data.get('aqi', 3)
        greenery = edge_data.get('greenery_score', 0)
        distance = edge_data.get('length', 1)

        reward = -aqi + 0.1 * greenery - 0.05 * distance

        self.current_node = chosen
        done = self.current_node == self.end_node

        return self.get_state(), reward, done

# --- Weather-Aware Environment ---
class WeatherEnv(Env):
    def __init__(self, graph, start_node, end_node, weather_factors):
        super().__init__(graph, start_node, end_node)
        self.weather_factors = weather_factors

    def step(self, action):
        neighbors = list(self.graph.neighbors(self.current_node))

        if len(neighbors) == 0:
            return self.get_state(), -100, True

        if action >= len(neighbors):
            chosen = neighbors[0]
        else:
            chosen = neighbors[action]

        edge_data = self.graph.get_edge_data(self.current_node, chosen)
        if isinstance(edge_data, dict) and 0 in edge_data:
            edge_data = edge_data[0]

        aqi = edge_data.get('aqi', 3)
        greenery = edge_data.get('greenery_score', 0)
        distance = edge_data.get('length', 1)

        weather_adjusted_aqi = aqi * self.weather_factors.get('pollution_multiplier', 1.0)
        weather_greenery_bonus = greenery * self.weather_factors.get('heat_penalty', 1.0)

        reward = (
            -weather_adjusted_aqi * 2.0 +
            weather_greenery_bonus * 0.15 +
            -distance * 0.05 +
            self.weather_factors.get('comfort_score', 5.0) * 0.1
        )

        self.current_node = chosen
        done = self.current_node == self.end_node

        return self.get_state(), reward, done


# --- RL Training Function (GPU/CPU Aware) ---
def train_rl_agent(graph, start_node, end_node, weather_factors=None, episodes=400):
    """
    Train an RL agent using either the base Env or WeatherEnv.
    Returns the trained agent instance.
    """
    # 1. DETERMINE DEVICE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}") 

    if weather_factors:
        env = WeatherEnv(graph, start_node, end_node, weather_factors)
        print(f"Training weather-aware RL agent for {episodes} episodes...")
    else:
        env = Env(graph, start_node, end_node)
        print(f"Training base RL agent for {episodes} episodes...")

    state_dim = 15
    action_dim = 5
    
    # 2. MOVE MODEL TO DEVICE
    agent = DQNAgent(state_dim, action_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    gamma = 0.9
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.1
    memory = []

    for episode in range(episodes):
        env.current_node = start_node
        total_reward = 0
        done = False
        
        visited_in_episode = set([start_node]) 

        while not done:
            features, neighbors = env.get_state()
            
            # 3. MOVE INPUT STATE TO DEVICE FOR INFERENCE/ACTION SELECTION
            state = torch.FloatTensor(features.flatten()).unsqueeze(0).to(device) 

            if random.random() < epsilon:
                action_index = random.randint(0, len(neighbors) - 1) if neighbors else 0
            else:
                with torch.no_grad():
                    q_values = agent(state)
                    valid_q_values = q_values[0, :len(neighbors)]
                    action_index = torch.argmax(valid_q_values).item()

            (next_features, next_neighbors), reward, done = env.step(action_index)
            total_reward += reward

            # NOTE: Store tensors on CPU to manage VRAM usage in the replay buffer
            next_state_tensor = torch.FloatTensor(next_features.flatten()).unsqueeze(0)
            memory.append((state.cpu(), action_index, reward, next_state_tensor, done))

            if len(memory) > 1000:
                memory.pop(0)

            # Training step
            if len(memory) >= 32:
                batch = random.sample(memory, 32)
                
                # 4. MOVE BATCH TENSORS TO DEVICE FOR TRAINING
                states_batch = torch.cat([x[0] for x in batch]).to(device)
                actions_batch = torch.LongTensor([x[1] for x in batch]).unsqueeze(1).to(device)
                rewards_batch = torch.FloatTensor([x[2] for x in batch]).unsqueeze(1).to(device)
                next_states_batch = torch.cat([x[3] for x in batch]).to(device)
                dones_batch = torch.BoolTensor([x[4] for x in batch]).unsqueeze(1).to(device)

                current_q = agent(states_batch).gather(1, actions_batch)
                max_next_q = agent(next_states_batch).max(1)[0].unsqueeze(1)
                expected_q = rewards_batch + gamma * max_next_q * (~dones_batch)

                loss = criterion(current_q, expected_q.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        if episode % 100 == 0:
            print(f"  Episode {episode}: Reward={total_reward:.1f}, Epsilon={epsilon:.2f}")

    print("RL training complete!")
    return agent

# --- RL Route Generation Function (GPU/CPU Aware) ---
def get_rl_route(agent, graph, start_node, end_node, max_steps=100):
    """
    Uses the trained DQN agent to find the optimal route.
    """
    # Use the base Env class for route generation logic
    env = Env(graph, start_node, end_node)
    
    # 1. DETERMINE AGENT'S DEVICE
    # Get the device the agent is currently on (cpu or cuda)
    device = next(agent.parameters()).device 
    
    env.current_node = env.start_node
    route = [env.start_node]
    done = False
    
    # Temporarily set the agent to evaluation mode
    agent.eval() 
    
    for _ in range(max_steps):
        if done:
            break
            
        features, neighbors = env.get_state()
        
        if not neighbors:
            print("Route stopped: Dead end reached.")
            break

        state = torch.FloatTensor(features.flatten()).unsqueeze(0)
        
        # 2. MOVE INPUT STATE TO AGENT'S DEVICE
        state = state.to(device) 
        
        with torch.no_grad():
            # Get Q-values and select the action with the highest value
            q_values = agent(state)
            valid_q_values = q_values[0, :len(neighbors)]
            action_index = torch.argmax(valid_q_values).item()

        # Execute the chosen action
        # Note: env.step uses numpy/python logic, which is fine on CPU
        (_, _), _, done = env.step(action_index) 
        
        chosen_node = env.current_node
        route.append(chosen_node)

        if chosen_node == env.end_node:
            break
            
    # Set the agent back to training mode
    agent.train() 
    return route