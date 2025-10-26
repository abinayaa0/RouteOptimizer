# visualization.py

import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import pandas as pd
from routing import calculate_route_metrics # Import the necessary utility

# --- Function Definitions (from original script) ---

def plot_final_maps(graph, place_name, food_pois, toilet_pois, run_pollution_analysis):
    # ... (body of the original plot_final_maps function) ...
    pass # (The body remains the same, using imported libraries)
    print("\nPlotting Final Results...")

    if run_pollution_analysis:
        aqi_values = [data.get('aqi') for u, v, data in graph.edges(data=True) if data.get('aqi') is not None]
        if not aqi_values:
            print("Could not plot Air Quality map. No AQI data found on the graph.")
        else:
            print("Plotting Air Quality map...")
            edge_colors_aqi = []
            for u, v, data in graph.edges(data=True):
                aqi = data.get('aqi')
                if aqi == 1: color = '#00FF00'
                elif aqi == 2: color = '#FFFF00'
                elif aqi == 3: color = '#FFA500'
                elif aqi == 4: color = '#FF0000'
                elif aqi == 5: color = '#800080'
                else: color = '#808080'
                edge_colors_aqi.append(color)
            fig, ax = ox.plot_graph(graph, node_size=0, edge_linewidth=1.2, edge_color=edge_colors_aqi, bgcolor='#333333', show=False, close=False)
            legend_elements = [plt.Line2D([0], [0], color='#00FF00', lw=4, label='AQI 1: Good'), plt.Line2D([0], [0], color='#FFFF00', lw=4, label='AQI 2: Fair'), plt.Line2D([0], [0], color='#FFA500', lw=4, label='AQI 3: Moderate'), plt.Line2D([0], [0], color='#FF0000', lw=4, label='AQI 4: Poor'), plt.Line2D([0], [0], color='#800080', lw=4, label='AQI 5: Very Poor')]
            ax.legend(handles=legend_elements, loc='lower right', facecolor='white', framealpha=0.9)
            ax.set_title(f"Air Quality in {place_name}", fontsize=16, color='w')
            plt.show()


    scores = [data.get('greenery_score', 0) for u, v, data in graph.edges(data=True)]
    if not scores or max(scores) == 0:
        print("No greenery data to plot.")
    else:
        print("Plotting Satellite Greenery map with POIs...")
        max_score = max(scores)
        cmap = plt.get_cmap('YlGn')
        edge_colors_green = [cmap(data.get('greenery_score', 0) / max_score) for u, v, data in graph.edges(data=True)]
        fig, ax = ox.plot_graph(graph, node_size=0, edge_linewidth=1.2, edge_color=edge_colors_green, bgcolor='#333333', show=False, close=False)

        if food_pois is not None and not food_pois.empty:
            food_pois.plot(ax=ax, color='orange', markersize=50, marker='o', label='Food/Shops')
        if toilet_pois is not None and not toilet_pois.empty:
            toilet_pois.plot(ax=ax, color='lightblue', markersize=50, marker='s', label='Toilets')

        if (food_pois is not None and not food_pois.empty) or (toilet_pois is not None and not toilet_pois.empty):
            ax.legend(loc='lower right', facecolor='white', framealpha=0.9)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max_score))
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05, shrink=0.5)
        ticks = np.linspace(0, max_score, 5)
        cb.set_ticks(ticks)
        cb.set_ticklabels([f'{int(t)}%' for t in ticks])
        cb.set_label('Greenery Score (%)', color='white')
        cb.ax.xaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cb.ax.axes, 'xticklabels'), color='white')

        ax.set_title(f"Satellite Greenery & POIs in {place_name}", fontsize=16, color='w')
        plt.show()

# (Include plot_route_comparison_subplot, plot_distance_comparison, plot_environmental_comparison, 
#  plot_route_efficiency, plot_health_impact, create_comparison_dashboard, and create_weather_impact_chart here.
#  The bodies of these functions remain the same as in the original script, but they need to import 
#  calculate_route_metrics from 'routing' and use it.)
# ... (all other plotting functions) ...

def plot_route_comparison_subplot(ax, graph, routes, route_names, place_name):
    # ... (original body) ...
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

    ox.plot_graph(graph, ax=ax, node_size=0, edge_linewidth=0.3,
                  edge_color='lightgray', bgcolor='white', show=False)

    for i, (route, name) in enumerate(zip(routes, route_names)):
        if len(route) > 1:
            route_coords = [(graph.nodes[node]['x'], graph.nodes[node]['y']) for node in route]
            x_coords, y_coords = zip(*route_coords)

            ax.plot(x_coords, y_coords, color=colors[i % len(colors)],
                    linewidth=4, alpha=0.8, label=name)

            if i == 0:
                ax.scatter(x_coords[0], y_coords[0], c='green', s=150,
                          marker='o', zorder=5, label='Start', edgecolor='white', linewidth=2)
                ax.scatter(x_coords[-1], y_coords[-1], c='red', s=150,
                          marker='s', zorder=5, label='End', edgecolor='white', linewidth=2)

    ax.set_title(f"Route Comparison: {place_name}", fontsize=14, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

def plot_distance_comparison(ax, graph, routes, route_names):
    # ... (original body) ...
    distances = []
    names = []

    for route, name in zip(routes, route_names):
        if route:
            metrics = calculate_route_metrics(graph, route)
            distances.append(metrics.get('distance', 0))
            names.append(name.replace(' ', '\n'))

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = ax.bar(names, distances, color=colors[:len(names)])

    ax.set_title('Distance Comparison', fontweight='bold')
    ax.set_ylabel('Distance (meters)')

    for bar, distance in zip(bars, distances):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{int(distance)}m', ha='center', va='bottom', fontweight='bold')
                
def plot_environmental_comparison(ax, graph, routes, route_names):
    # ... (original body) ...
    aqi_scores = []
    greenery_scores = []
    names = []

    for route, name in zip(routes, route_names):
        if route:
            metrics = calculate_route_metrics(graph, route)
            aqi_scores.append(metrics.get('avg_aqi', 0))
            greenery_scores.append(metrics.get('avg_greenery', 0))
            names.append(name.replace(' ', '\n'))

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width/2, aqi_scores, width, label='Avg AQI', color='#FF7675', alpha=0.8)
    bars2 = ax.bar(x + width/2, greenery_scores, width, label='Avg Greenery %', color='#00B894', alpha=0.8)

    ax.set_title('Environmental Impact', fontweight='bold')
    ax.set_xlabel('Routes')
    ax.set_ylabel('Score')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)

def plot_route_efficiency(ax, graph, routes, route_names):
    # ... (original body) ...
    for i, (route, name) in enumerate(zip(routes, route_names)):
        if route:
            metrics = calculate_route_metrics(graph, route)
            distance = metrics.get('distance', 0)
            eco_score = 5 - metrics.get('avg_aqi', 3) + metrics.get('avg_greenery', 0) / 10

            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
            ax.scatter(distance, eco_score, s=200, color=colors[i % len(colors)],
                      alpha=0.7, edgecolor='white', linewidth=2, label=name)

    ax.set_title('Route Efficiency', fontweight='bold')
    ax.set_xlabel('Distance (meters)')
    ax.set_ylabel('Eco-Score')
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_health_impact(ax, graph, routes, route_names):
    # ... (original body) ...
    health_scores = []
    names = []

    for route, name in zip(routes, route_names):
        if route:
            metrics = calculate_route_metrics(graph, route)
            aqi_penalty = metrics.get('avg_aqi', 3) * 2
            greenery_bonus = metrics.get('avg_greenery', 0) * 0.1
            health_score = max(0, 10 - aqi_penalty + greenery_bonus)

            health_scores.append(health_score)
            names.append(name.replace(' ', '\n'))

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = ax.bar(names, health_scores, color=colors[:len(names)])

    ax.set_title('Estimated Health Impact', fontweight='bold')
    ax.set_ylabel('Health Score (0-10)')
    ax.set_ylim(0, 10)

    for bar, score in zip(bars, health_scores):
        if score < 4: bar.set_color('#FF6B6B')
        elif score < 7: bar.set_color('#F39C12')
        else: bar.set_color('#00B894')

        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{score:.1f}', ha='center', va='bottom', fontweight='bold')

def create_comparison_dashboard(graph, routes, route_names, place_name):
    """Create a comprehensive dashboard comparing all routes."""
    fig = plt.figure(figsize=(20, 15))

    ax1 = plt.subplot(2, 3, (1, 4))
    plot_route_comparison_subplot(ax1, graph, routes, route_names, place_name)

    ax2 = plt.subplot(2, 3, 2)
    plot_distance_comparison(ax2, graph, routes, route_names)

    ax3 = plt.subplot(2, 3, 3)
    plot_environmental_comparison(ax3, graph, routes, route_names)

    ax4 = plt.subplot(2, 3, 5)
    plot_route_efficiency(ax4, graph, routes, route_names)

    ax5 = plt.subplot(2, 3, 6)
    plot_health_impact(ax5, graph, routes, route_names)

    plt.tight_layout()
    plt.show()

def create_weather_impact_chart(weather_factors, routes, route_names, graph):
    """Create a chart showing weather impact on different routes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    factors = list(weather_factors.keys())
    values = list(weather_factors.values())
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    bars = ax1.barh(factors, values, color=colors)
    ax1.set_title('Current Weather Impact Factors', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Impact Multiplier')

    for bar, value in zip(bars, values):
        width = bar.get_width()
        ax1.text(width + 0.05, bar.get_y() + bar.get_height()/2,
                 f'{value:.2f}', ha='left', va='center', fontweight='bold')

    route_scores = []
    for route in routes:
        if route:
            metrics = calculate_route_metrics(graph, route)
            base_score = 10 - metrics.get('avg_aqi', 3) * 2 + metrics.get('avg_greenery', 0) * 0.1
            weather_score = base_score * (weather_factors.get('comfort_score', 5.0) / 5.0)
            route_scores.append(max(0, weather_score))

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    bars = ax2.bar([name.replace(' ', '\n') for name in route_names], route_scores,
                    color=colors[:len(route_scores)])

    ax2.set_title('Weather-Adjusted Route Quality', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Quality Score (0-10)')
    ax2.set_ylim(0, 10)

    for bar, score in zip(bars, route_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                 f'{score:.1f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()