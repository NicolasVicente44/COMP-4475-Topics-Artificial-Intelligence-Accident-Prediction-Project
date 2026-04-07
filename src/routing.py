# src/routing.py - A* pathfinding on the collision risk grid.

import heapq
import math
import numpy as np
import pandas as pd
from collections import defaultdict
from config import SHORELINE_LONS, SHORELINE_LATS


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance between two lat/lon points in meters."""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class RiskGrid:
    """Graph built from collision data for A* pathfinding. Each ~500m cell is a
    node connected to its 8 neighbors. Cells over Lake Ontario are impassable."""

    def __init__(self, grid_csv_path, grid_size=0.005):
        self.grid_size = grid_size
        self.cells = {}
        self.cell_data = {}
        self.default_risk = 0.05

        df = pd.read_csv(grid_csv_path)
        for _, row in df.iterrows():
            key = (round(row['glat'], 4), round(row['glon'], 4))
            self.cells[key] = row['route_risk']
            self.cell_data[key] = {
                'risk': row['route_risk'],
                'count': int(row['count']),
                'fatals': int(row['fatals']),
            }

        # Routable = has data OR within 2 steps of data (allows residential routing)
        self.routable = set(self.cells.keys())
        for key in list(self.cells.keys()):
            lat, lon = key
            for dlat in range(-2, 3):
                for dlon in range(-2, 3):
                    neighbor = (round(lat + dlat * grid_size, 4),
                                round(lon + dlon * grid_size, 4))
                    if not self._is_water(neighbor[0], neighbor[1]):
                        self.routable.add(neighbor)

        lats = [k[0] for k in self.cells]
        lons = [k[1] for k in self.cells]
        self.lat_min = min(lats) - grid_size * 5
        self.lat_max = max(lats) + grid_size * 5
        self.lon_min = min(lons) - grid_size * 5
        self.lon_max = max(lons) + grid_size * 5

        print(f"  RiskGrid: {len(self.cells)} data cells | {len(self.routable)} routable cells")
        print(f"  Bounds: ({self.lat_min:.3f},{self.lon_min:.3f}) to ({self.lat_max:.3f},{self.lon_max:.3f})")

    def _is_water(self, lat, lon):
        """Check if a point is south of the shoreline (in Lake Ontario)."""
        lons, lats = SHORELINE_LONS, SHORELINE_LATS
        if lon <= lons[0]:
            return lat < lats[0]
        if lon >= lons[-1]:
            return lat < lats[-1]
        for i in range(len(lons) - 1):
            if lons[i] <= lon <= lons[i + 1]:
                frac = (lon - lons[i]) / (lons[i + 1] - lons[i])
                shore_lat = lats[i] + frac * (lats[i + 1] - lats[i])
                return lat < shore_lat
        return False

    def snap_to_grid(self, lat, lon):
        """Snap a coordinate to the nearest grid point."""
        glat = round(round(lat / self.grid_size) * self.grid_size, 4)
        glon = round(round(lon / self.grid_size) * self.grid_size, 4)
        return (glat, glon)

    def get_risk(self, node):
        return self.cells.get(node, self.default_risk)

    def is_routable(self, node):
        return node in self.routable

    def in_bounds(self, node):
        return (self.lat_min <= node[0] <= self.lat_max and
                self.lon_min <= node[1] <= self.lon_max)

    def neighbors(self, node):
        """Return 8-connected neighbors that are on land."""
        lat, lon = node
        gs = self.grid_size
        dirs = [
            (gs, 0), (-gs, 0), (0, gs), (0, -gs),
            (gs, gs), (gs, -gs), (-gs, gs), (-gs, -gs)
        ]
        result = []
        for dlat, dlon in dirs:
            nb = (round(lat + dlat, 4), round(lon + dlon, 4))
            if self.in_bounds(nb) and self.is_routable(nb):
                result.append(nb)
        return result


def heuristic_distance(a, b):
    """Haversine heuristic for A* (km)."""
    return haversine(a[0], a[1], b[0], b[1]) / 1000.0


def heuristic_risk(a, b, grid):
    """Admissible risk heuristic: straight-line cells * min possible risk."""
    cells_remaining = heuristic_distance(a, b) / (grid.grid_size * 111)
    return cells_remaining * grid.default_risk * 0.5


def astar(grid, start, goal, mode="safest"):
    """A* search on the risk grid. Mode is 'safest' or 'shortest'."""
    start = grid.snap_to_grid(*start)
    goal = grid.snap_to_grid(*goal)

    open_set = [(0, 0, start)]
    came_from = {}
    g_score = defaultdict(lambda: float('inf'))
    g_score[start] = 0
    risk_accumulated = defaultdict(lambda: 0.0)
    dist_accumulated = defaultdict(lambda: 0.0)
    counter = 1
    explored = 0

    while open_set:
        current_f, _, current = heapq.heappop(open_set)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return {
                'path': path,
                'total_cost': g_score[goal],
                'risk_sum': risk_accumulated[goal],
                'distance_km': dist_accumulated[goal],
                'explored': explored,
            }

        explored += 1
        if explored > 50000:
            print(f"  WARNING: A* hit {explored} node limit.")
            break

        for nb in grid.neighbors(current):
            d = haversine(current[0], current[1], nb[0], nb[1]) / 1000.0

            if mode == "safest":
                risk = grid.get_risk(nb)
                step_cost = risk * d
            else:
                step_cost = d

            tentative_g = g_score[current] + step_cost

            if tentative_g < g_score[nb]:
                came_from[nb] = current
                g_score[nb] = tentative_g
                risk_accumulated[nb] = risk_accumulated[current] + grid.get_risk(nb) * d
                dist_accumulated[nb] = dist_accumulated[current] + d

                if mode == "safest":
                    h = heuristic_risk(nb, goal, grid)
                else:
                    h = heuristic_distance(nb, goal)

                heapq.heappush(open_set, (tentative_g + h, counter, nb))
                counter += 1

    return None


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
        'risk_reduction': 1.0 - (safest['risk_sum'] / max(shortest['risk_sum'], 0.001)),
        'distance_increase': (safest['distance_km'] / max(shortest['distance_km'], 0.001)) - 1.0,
    }

    print(f"  Shortest: {shortest['distance_km']:.2f} km | Risk: {shortest['risk_sum']:.4f} | Nodes: {len(shortest['path'])}")
    print(f"  Safest:   {safest['distance_km']:.2f} km | Risk: {safest['risk_sum']:.4f} | Nodes: {len(safest['path'])}")
    print(f"  Risk reduction: {result['risk_reduction']*100:.1f}% | Extra distance: {result['distance_increase']*100:.1f}%")

    return result


if __name__ == "__main__":
    grid = RiskGrid("outputs/risk_grid.csv")

    routes = [
        ("Downtown to Scarborough",  (43.6550, -79.3830), (43.7730, -79.2580)),
        ("Etobicoke to East York",   (43.6440, -79.5100), (43.6920, -79.3270)),
        ("North York to Waterfront",  (43.7670, -79.4110), (43.6390, -79.3810)),
    ]

    for name, start, goal in routes:
        print(f"\n{'='*50}")
        print(f"  {name}")
        result = compare_routes(grid, start, goal)