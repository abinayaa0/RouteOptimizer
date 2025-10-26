# main.py

import osmnx as ox
from dotenv import load_dotenv


# Import functions/data from your modules
from config import LOCATION, NETWORK_DISTANCE, OPENWEATHER_API_KEY, RUN_POLLUTION_ANALYSIS, RUN_WEATHER_FETCH
from data_fetch import (
    get_current_weather, get_air_pollution_data, get_satellite_greenery_data, 
    get_pois, get_weather_routing_factors, add_sample_environmental_data
)
from rl_agent import train_rl_agent
from routing import (
    get_dijkstra_route, get_eco_dijkstra_route, get_weather_aware_eco_route, 
    get_rl_route, calculate_route_metrics
)
from visualization import (
    create_comparison_dashboard, create_weather_impact_chart, plot_final_maps
)

#Load environment variables from .env file
load_dotenv()

def main_execution(new_endpoint=None):
    """
    Main function to run the full GreenPaths analysis.
    Optionally accepts a new end node for longer routes without retraining.
    """
    print("🌍 GREENPATHS ANALYSIS: Initializing...")
    
    # --- Step 1: Download Base Map ---
    print(f"\n1. Downloading street network for: {LOCATION}")
    G = ox.graph_from_address(LOCATION, dist=NETWORK_DISTANCE, network_type='walk')
    G_enriched = G.copy()
    
    # --- Step 2: Fetch Weather Data ---
    weather_data, weather_factors = None, {'heat_penalty': 1.0, 'pollution_multiplier': 1.0, 'comfort_score': 5.0}
    if RUN_WEATHER_FETCH and OPENWEATHER_API_KEY != "YOUR_API_KEY":
        weather_data, weather_factors = get_weather_routing_factors(LOCATION, OPENWEATHER_API_KEY)

    # --- Step 3: Enrich Graph with Environmental Data ---
    if OPENWEATHER_API_KEY != "YOUR_API_KEY":
        # G_enriched = get_air_pollution_data(G_enriched, OPENWEATHER_API_KEY) # Disabled for speed/rate limits
        G_enriched = add_sample_environmental_data(G_enriched) # Use sample data for demo/speed
        G_enriched = get_satellite_greenery_data(G_enriched)
    else:
        print("\nUsing sample environmental data (API key not set).")
        G_enriched = add_sample_environmental_data(G_enriched)

    # --- Step 4: Define Route Endpoints ---
    nodes = list(G_enriched.nodes())
    start_node = nodes[len(nodes)//4]
    
    if new_endpoint:
        # Use the provided new endpoint (for longer paths)
        end_node = new_endpoint
        print(f"**Using NEW, longer end node: {end_node}**")
    else:
        # Use the original default end node
        end_node = nodes[3*len(nodes)//4]
        print(f"Using default end node: {end_node}")

    # --- Step 5: Train/Reuse RL Agent ---
    # We only train if a new endpoint isn't specified, or if we need a fresh start.
    # To reuse the agent for a longer route, we need to pass the trained agent object.
    
    # The simplest implementation for a standalone run is to train every time:
    print("\n3. Training Weather-Aware AI agent...")
    weather_rl_agent = train_rl_agent(G_enriched, start_node, end_node, weather_factors, episodes=300)

    # --- Step 6: Generate Routes ---
    print("\n4. Generating comparison routes...")
    dijkstra_route = get_dijkstra_route(G_enriched, start_node, end_node)
    eco_dijkstra_route = get_eco_dijkstra_route(G_enriched, start_node, end_node)
    weather_eco_route = get_weather_aware_eco_route(G_enriched, start_node, end_node, weather_factors)
    weather_rl_route = get_rl_route(weather_rl_agent, G_enriched, start_node, end_node)

    routes = [dijkstra_route, eco_dijkstra_route, weather_eco_route, weather_rl_route]
    route_names = ['Standard Shortest', 'Eco-Dijkstra', 'Weather-Aware Eco', 'Weather-Aware AI']

    # --- Step 7: Analyze and Visualize ---
    valid_routes = [route for route in routes if route]
    valid_names = [name for route, name in zip(routes, route_names) if route]

    if valid_routes:
        print("\n5. Generating Dashboard and Visualizations...")
        create_comparison_dashboard(G_enriched, valid_routes, valid_names, LOCATION)
        create_weather_impact_chart(weather_factors, valid_routes, valid_names, G_enriched)
        
    food_locations, toilet_locations = get_pois(LOCATION)
    plot_final_maps(G_enriched, LOCATION, food_locations, toilet_locations, RUN_POLLUTION_ANALYSIS)

    return weather_rl_agent, G_enriched, routes, route_names, weather_factors

def run_longer_route_scenario(agent, graph, old_routes, old_names, weather_factors):
    """
    Re-runs the routing with a new, further endpoint using the EXISTING agent.
    """
    print("\n\n" + "="*80)
    print("🚀 SCENARIO: FINDING LONGER PATH (REUSING EXISTING AGENT)")
    print("="*80)
    
    nodes = list(graph.nodes())
    start_node = old_routes[0][0] # Use start node from previous run
    new_end_node = nodes[len(nodes) - 5] # Select a node near the end for a longer route

    print(f"Finding NEW routes from node {start_node} to further node {new_end_node}")

    # Re-generate all routes (Dijkstra and Weather-Aware Eco don't rely on the RL agent)
    new_dijkstra_route = get_dijkstra_route(graph, start_node, new_end_node)
    new_eco_dijkstra_route = get_eco_dijkstra_route(graph, start_node, new_end_node)
    new_weather_eco_route = get_weather_aware_eco_route(graph, start_node, new_end_node, weather_factors)
    
    # Use the EXISTING agent for the new, longer route (as requested)
    new_weather_rl_route = get_rl_route(agent, graph, start_node, new_end_node)

    new_routes = [new_dijkstra_route, new_eco_dijkstra_route, new_weather_eco_route, new_weather_rl_route]
    route_names = ['Standard Shortest', 'Eco-Dijkstra', 'Weather-Aware Eco', 'Weather-Aware AI']

    # Print analysis
    valid_new_routes = []
    valid_new_names = []
    
    for route, name in zip(new_routes, route_names):
        if route:
            metrics = calculate_route_metrics(graph, route)
            weather_health = max(0, 10 - metrics.get('avg_aqi', 3) * 2 + metrics.get('avg_greenery', 0) * 0.1) * (weather_factors.get('comfort_score', 5.0) / 5.0)

            print(f"\n🛤️ {name.upper()}:")
            print(f"    📏 Distance: {metrics.get('distance', 0):.0f}m")
            print(f"    💚 Health Score: {weather_health:.1f}/10")
            
            valid_new_routes.append(route)
            valid_new_names.append(name)

    # Visualize the new longer routes
    if valid_new_routes:
        create_comparison_dashboard(graph, valid_new_routes, valid_new_names, LOCATION + " (Longer Path)")

if __name__ == '__main__':
    # --- Run Initial Scenario (Default Endpoint) ---
    trained_agent, G_enriched, initial_routes, initial_names, weather_factors = main_execution()
    
    # --- Run Longer Route Scenario (New Endpoint) ---
    run_longer_route_scenario(trained_agent, G_enriched, initial_routes, initial_names, weather_factors)