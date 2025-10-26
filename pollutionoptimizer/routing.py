# routing.py

import networkx as nx
import torch
from rl_agent import get_rl_route, Env # Import Env for RL route generation
def get_dijkstra_route(graph, start_node, end_node, weight='length'):
    # ... (body of the original get_dijkstra_route function) ...
    try:
        route = nx.shortest_path(graph, start_node, end_node, weight=weight)
        return route
    except nx.NetworkXNoPath:
        print(f"No path found between {start_node} and {end_node}.")
        return []

def get_eco_dijkstra_route(graph, start_node, end_node):
    # ... (body of the original get_eco_dijkstra_route function) ...
    G_eco = graph.copy()

    for u, v, key, data in G_eco.edges(keys=True, data=True):
        base_length = data.get('length', 100)
        aqi = data.get('aqi', 3)
        greenery = data.get('greenery_score', 0)

        eco_weight = base_length * (1 + aqi * 0.5) * (1 - min(greenery, 50) * 0.01)
        G_eco.edges[u, v, key]['eco_weight'] = eco_weight

    try:
        route = nx.shortest_path(G_eco, start_node, end_node, weight='eco_weight')
        return route
    except nx.NetworkXNoPath:
        print("No eco-friendly path found between start and end nodes.")
        return []

def get_weather_aware_eco_route(graph, start_node, end_node, weather_factors):
    # ... (body of the original get_weather_aware_eco_route function) ...
    G_weather = graph.copy()

    for u, v, key, data in G_weather.edges(keys=True, data=True):
        base_length = data.get('length', 100)
        aqi = data.get('aqi', 3)
        greenery = data.get('greenery_score', 0)

        weather_aqi = aqi * weather_factors.get('pollution_multiplier', 1.0)
        weather_greenery = greenery * weather_factors.get('heat_penalty', 1.0)

        eco_weight = (
            base_length *
            (1 + weather_aqi * 0.6) *
            (1 - min(weather_greenery, 60) * 0.015) *
            (2.0 - weather_factors.get('comfort_score', 5.0) * 0.2)
        )

        G_weather.edges[u, v, key]['weather_eco_weight'] = max(eco_weight, base_length * 0.1)

    try:
        route = nx.shortest_path(G_weather, start_node, end_node, weight='weather_eco_weight')
        return route
    except nx.NetworkXNoPath:
        print("No weather-aware eco path found.")
        return []


def get_rl_route(agent, graph, start_node, end_node, max_steps=100):
    # ... (body of the original get_rl_route function) ...
    """Generate a route using the trained RL agent."""
    # Use the base Env class for route generation, as it only needs step/get_state.
    env = Env(graph, start_node, end_node) 
    route = [start_node]
    current_node = start_node
    visited = set([start_node])

    for step in range(max_steps):
        if current_node == end_node:
            break

        features, neighbors = env.get_state()
        state = torch.FloatTensor(features.flatten()).unsqueeze(0)

        # Get valid unvisited neighbors
        valid_neighbors = [n for n in neighbors if n not in visited]
        if not valid_neighbors:
            valid_neighbors = neighbors

        if not valid_neighbors:
            break

        with torch.no_grad():
            q_values = agent(state)

        valid_actions = []
        for i, neighbor in enumerate(neighbors):
            if neighbor in valid_neighbors:
                valid_actions.append(i)

        if valid_actions:
            best_action_idx = max(valid_actions, key=lambda i: q_values[0, i].item())
            chosen_neighbor = neighbors[best_action_idx]
        else:
            chosen_neighbor = neighbors[0]

        route.append(chosen_neighbor)
        visited.add(chosen_neighbor)
        current_node = chosen_neighbor
        env.current_node = current_node

    return route

def calculate_route_metrics(graph, route):
    # ... (body of the original calculate_route_metrics function) ...
    if len(route) < 2:
        return {}

    total_distance = 0
    total_aqi = 0
    total_greenery = 0
    edge_count = 0

    for i in range(len(route) - 1):
        u, v = route[i], route[i + 1]
        edge_data = graph.get_edge_data(u, v)
        if edge_data:
            if isinstance(edge_data, dict) and 0 in edge_data:
                edge_data = edge_data[0]

            total_distance += edge_data.get('length', 0)
            aqi = edge_data.get('aqi')
            if aqi is not None:
                total_aqi += aqi
                edge_count += 1
            total_greenery += edge_data.get('greenery_score', 0)

    avg_aqi = total_aqi / edge_count if edge_count > 0 else 0
    avg_greenery = total_greenery / (len(route) - 1) if len(route) > 1 else 0

    return {
        'distance': total_distance,
        'avg_aqi': avg_aqi,
        'avg_greenery': avg_greenery,
        'steps': len(route) - 1
    }