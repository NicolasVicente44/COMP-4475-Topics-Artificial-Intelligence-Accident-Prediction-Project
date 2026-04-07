# src/routing.py - OpenStreetMap pathfinding with risk.

import os
import networkx as nx
import osmnx as ox
import pandas as pd
from scipy.spatial import KDTree
import warnings

# Suppress osmnx/pandas warnings
warnings.filterwarnings("ignore")

class RiskGrid:
    """Graph built from OSMnx data for A* pathfinding.
    Risk values from the predictive model are assigned to edges."""

    def __init__(self, grid_csv_path):
        self.grid_csv_path = grid_csv_path
        self.default_risk = 0.05
        
        # Load risk data
        self.df = pd.read_csv(grid_csv_path)
        self.tree = KDTree(self.df[['glat', 'glon']].values)
        self.risks = self.df['route_risk'].values

        graph_path = grid_csv_path.replace(".csv", "_osmnx.graphml")
        if os.path.exists(graph_path):
            print(f"  RiskGrid: Loading cached OSM graph from {graph_path}...")
            self.G = ox.load_graphml(graph_path)
        else:
            print("  RiskGrid: Downloading OSM graph for Toronto & Mississauga (this may take a few minutes)...")
            places = ["Toronto, Ontario, Canada", "Mississauga, Ontario, Canada"]
            self.G = ox.graph_from_place(places, network_type="drive", simplify=True)
            
            print(f"  RiskGrid: Saving OSM graph to {graph_path}...")
            ox.save_graphml(self.G, graph_path)
            
        self._assign_risks()
        print(f"  RiskGrid: Ready with {len(self.G.nodes)} nodes and {len(self.G.edges)} edges.")

        
    def _assign_risks(self):
        """Assign distance and risk weights to each edge."""
        # Process node coordinates once for fast access
        node_coords = {n: (data['y'], data['x']) for n, data in self.G.nodes(data=True)}
        
        # To avoid zero cost exploiting
        min_risk = 0.05
        
        for u, v, k, data in self.G.edges(keys=True, data=True):
            # Distance in km (length is in meters from OSMnx)
            dist_km = data.get('length', 0) / 1000.0
            data['distance_km'] = dist_km
            
            # Midpoint to sample risk
            u_lat, u_lon = node_coords[u]
            v_lat, v_lon = node_coords[v]
            mid_lat = (u_lat + v_lat) / 2
            mid_lon = (u_lon + v_lon) / 2
            
            _, idx = self.tree.query([[mid_lat, mid_lon]])
            risk = self.risks[idx[0]]
            risk = max(risk, min_risk)
            
            data['risk_val'] = risk
            data['risk_cost'] = risk * dist_km

def astar(grid, start, goal, mode="safest"):
    """Find safest or shortest route using OSMnx network and A* algorithm."""
    import math
    
    # Nearest nodes (osmnx expects X/lon, then Y/lat)
    source = ox.distance.nearest_nodes(grid.G, start[1], start[0])
    target = ox.distance.nearest_nodes(grid.G, goal[1], goal[0])
    
    weight = 'risk_cost' if mode == 'safest' else 'distance_km'
    
    def heuristic(u, v):
        """Haversine distance heuristic for A*"""
        u_y, u_x = grid.G.nodes[u]['y'], grid.G.nodes[u]['x']
        v_y, v_x = grid.G.nodes[v]['y'], grid.G.nodes[v]['x']
        
        R = 6371.0  # km
        dlat = math.radians(v_y - u_y)
        dlon = math.radians(v_x - u_x)
        a = math.sin(dlat / 2)**2 + math.cos(math.radians(u_y)) * math.cos(math.radians(v_y)) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        dist_km = R * c
        
        # If searching for safest route, scale the distance by the minimum possible risk 
        # to ensure the heuristic remains admissible (never overestimates the cost)
        if mode == "safest":
            return dist_km * 0.05
        return dist_km

    try:
        path_nodes = nx.astar_path(grid.G, source, target, heuristic=heuristic, weight=weight)
    except nx.NetworkXNoPath:
        return None
        
    path = [(grid.G.nodes[n]['y'], grid.G.nodes[n]['x']) for n in path_nodes]
    
    dist_sum = 0
    risk_sum = 0
    total_cost = 0
    
    for i in range(len(path_nodes)-1):
        u, v = path_nodes[i], path_nodes[i+1]
        edge_data = grid.G.get_edge_data(u, v)[0]
        
        d = edge_data.get('distance_km', 0)
        r = edge_data.get('risk_val', grid.default_risk)
        w = edge_data.get(weight, 0)
        
        dist_sum += d
        risk_sum += r * d
        total_cost += w
        
    return {
        'path': path,
        'total_cost': total_cost,
        'risk_sum': risk_sum,
        'distance_km': dist_sum,
        'explored': len(path_nodes), # compatibility metric
    }

def compare_routes(grid, start, goal):
    """Find both shortest and safest routes and return comparison stats."""
    print(f"\n  Routing from ({start[0]:.4f}, {start[1]:.4f}) to ({goal[0]:.4f}, {goal[1]:.4f})")

    shortest = astar(grid, start, goal, mode="shortest")
    safest = astar(grid, start, goal, mode="safest")

    if not shortest or not safest:
        print("  ERROR: Could not find a path!")
        return None

    result = {
        'start': start,
        'goal': goal,
        'shortest': shortest,
        'safest': safest,
        'risk_reduction': 1.0 - (safest['risk_sum'] / max(shortest['risk_sum'], 0.0001)),
        'distance_increase': (safest['distance_km'] / max(shortest['distance_km'], 0.0001)) - 1.0,
    }

    print(f"  Shortest: {shortest['distance_km']:.2f} km | Risk: {shortest['risk_sum']:.4f}")
    print(f"  Safest:   {safest['distance_km']:.2f} km | Risk: {safest['risk_sum']:.4f}")
    print(f"  Risk reduction: {result['risk_reduction']*100:.1f}% | Extra distance: {result['distance_increase']*100:.1f}%")

    return result