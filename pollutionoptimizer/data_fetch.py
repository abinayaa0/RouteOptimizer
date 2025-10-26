# data_fetch.py

import osmnx as ox
import requests
import time
import geopandas as gpd
import numpy as np
import contextily as cx
from config import OPENWEATHER_API_KEY, LOCATION

# --- Function Definitions (from original script) ---

def get_current_weather(address, api_key):
    # ... (body of the original get_current_weather function) ...
    # This function fetches current weather for a single location.
    print("\nFetching current weather...")
    try:
        point = ox.geocode(address)
        lat, lon = point[0], point[1]
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        response = requests.get(url)
        response.raise_for_status()
        weather_data = response.json()

        description = weather_data['weather'][0]['description']
        temp = weather_data['main']['temp']
        humidity = weather_data['main']['humidity']

        print(f"Weather in {address.split(',')[0]}: {description.title()}, {temp}°C, {humidity}% Humidity")
        return weather_data
    except Exception as e:
        print(f"Could not fetch weather data. Error: {e}")
        return None

def get_air_pollution_data(graph, api_key):
    # ... (body of the original get_air_pollution_data function) ...
    # This function enriches the graph with AQI data for each edge.
    print("\nStarting to fetch air pollution data...")
    edges = list(graph.edges(keys=True, data=True))
    total_edges = len(edges)
    for i, (u, v, key, data) in enumerate(edges):
        if 'geometry' in data:
            midpoint = data['geometry'].interpolate(0.5, normalized=True)
            lat, lon = midpoint.y, midpoint.x
        else:
            lat1, lon1 = graph.nodes[u]['y'], graph.nodes[u]['x']
            lat2, lon2 = graph.nodes[v]['y'], graph.nodes[v]['x']
            lat, lon = (lat1 + lat2) / 2, (lat1 + lon2) / 2

        url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"

        retries = 3
        for attempt in range(retries):
            try:
                response = requests.get(url)
                response.raise_for_status()
                pollution_data = response.json()['list'][0]
                aqi = pollution_data['main']['aqi']
                graph.edges[u, v, key]['aqi'] = aqi
                if (i + 1) % 10 == 0:
                    print(f"  ...processed {i+1}/{total_edges} streets for pollution.")
                break
            except requests.exceptions.RequestException as e:
                # ... (error handling from original script) ...
                if 'response' in locals() and response.status_code == 429 and attempt < retries - 1:
                    print("    Rate limit hit. Waiting for 30 seconds before retrying...")
                    time.sleep(30)
                else:
                    graph.edges[u, v, key]['aqi'] = None
                    break
        
        # Reduced sleep time for testing; increase for real usage if rate limits are hit
        if i % 10 == 0:
            time.sleep(0.5) 

    print("Finished fetching air pollution data.")
    return graph

def get_satellite_greenery_data(graph):
    # ... (body of the original get_satellite_greenery_data function) ...
    # This function enriches the graph with a greenery score.
    print("\nStarting to fetch and analyze satellite imagery for greenery...")
    edges = list(graph.edges(keys=True, data=True))
    total_edges = len(edges)
    for i, (u, v, key, data) in enumerate(edges):
        if 'geometry' not in data:
            graph.edges[u, v, key]['greenery_score'] = 0
            continue
        bounds = data['geometry'].buffer(20).bounds
        west, south, east, north = bounds
        try:
            image, extent = cx.bounds2img(west, south, east, north, source=cx.providers.Esri.WorldImagery, ll=True)
            img_array = np.array(image)
            green_mask = (img_array[:, :, 1] > img_array[:, :, 0]) & \
                         (img_array[:, :, 1] > img_array[:, :, 2]) & \
                         (img_array[:, :, 1] > 50)
            green_percentage = np.mean(green_mask) * 100
            graph.edges[u, v, key]['greenery_score'] = green_percentage
            if (i + 1) % 20 == 0:
                print(f"  ...analyzed satellite image for {i+1}/{total_edges} streets.")
        except Exception as e:
            graph.edges[u, v, key]['greenery_score'] = 0
    print("Finished analyzing satellite imagery.")
    return graph

def get_pois(address):
    # ... (body of the original get_pois function) ...
    # This function fetches Points of Interest (POIs).
    print("\nFetching Points of Interest (POIs)...")
    try:
        point = ox.geocode(address)
        tags_food = {"amenity": ["restaurant", "cafe", "fast_food", "food_court"], "shop": "convenience"}
        food_pois = ox.features_from_point(point, tags_food, dist=800)
        print(f"Found {len(food_pois)} food-related POIs.")

        tags_toilets = {"amenity": "toilets"}
        toilet_pois = ox.features_from_point(point, tags_toilets, dist=800)
        print(f"Found {len(toilet_pois)} public toilets.")

        return food_pois, toilet_pois
    except Exception as e:
        print(f"Could not fetch POIs. Error: {e}")
        return gpd.GeoDataFrame(), gpd.GeoDataFrame()

def get_weather_routing_factors(address, api_key):
    # ... (body of the original get_weather_routing_factors and calculate_weather_routing_impact) ...
    print("\nFetching weather data for routing decisions...")
    # (The two original functions are combined here for simplicity, or kept separate if preferred)
    
    # Keeping the original functions separated for modularity
    def calculate_weather_routing_impact(temp, humidity, wind_speed, description):
        # ... (original body of calculate_weather_routing_impact) ...
        factors = {}

        if temp > 35: factors['heat_penalty'] = 2.0; factors['comfort_score'] = 2.0
        elif temp > 30: factors['heat_penalty'] = 1.5; factors['comfort_score'] = 3.0
        elif temp < 15: factors['heat_penalty'] = 0.8; factors['comfort_score'] = 7.0
        else: factors['heat_penalty'] = 1.0; factors['comfort_score'] = 8.0

        if humidity > 80: factors['pollution_multiplier'] = 1.5; factors['comfort_score'] *= 0.8
        elif humidity > 60: factors['pollution_multiplier'] = 1.2; factors['comfort_score'] *= 0.9
        else: factors['pollution_multiplier'] = 1.0

        if wind_speed > 5: factors['pollution_multiplier'] *= 0.8; factors['comfort_score'] *= 1.1
        elif wind_speed < 1: factors['pollution_multiplier'] *= 1.3; factors['comfort_score'] *= 0.9

        if 'rain' in description.lower(): factors['comfort_score'] *= 0.6; factors['pollution_multiplier'] *= 0.7
        elif 'clear' in description.lower(): factors['comfort_score'] *= 1.1
        
        return factors


    try:
        point = ox.geocode(address)
        lat, lon = point[0], point[1]
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        response = requests.get(url)
        response.raise_for_status()
        weather_data = response.json()

        temp = weather_data['main']['temp']
        humidity = weather_data['main']['humidity']
        wind_speed = weather_data.get('wind', {}).get('speed', 0)
        description = weather_data['weather'][0]['description']

        weather_factors = calculate_weather_routing_impact(temp, humidity, wind_speed, description)

        print(f"Weather routing factors: Heat penalty={weather_factors['heat_penalty']:.1f}, "
              f"Pollution multiplier={weather_factors['pollution_multiplier']:.1f}")
        return weather_data, weather_factors

    except Exception as e:
        print(f"Could not fetch weather data. Using default factors. Error: {e}")
        return None, {'heat_penalty': 1.0, 'pollution_multiplier': 1.0, 'comfort_score': 5.0}

def add_sample_environmental_data(graph):
    """Utility to add sample data for testing when API is unavailable."""
    np.random.seed(42)
    for u, v, key, data in graph.edges(keys=True, data=True):
        graph.edges[u, v, key]['aqi'] = np.random.randint(1, 5)
        graph.edges[u, v, key]['greenery_score'] = np.random.uniform(0, 45)
    return graph