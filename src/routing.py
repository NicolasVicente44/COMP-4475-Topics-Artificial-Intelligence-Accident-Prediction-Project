"""
src/routing.py - A* pathfinding for risk-aware routing.

Builds a graph from the collision risk grid and finds paths that
minimize accident risk instead of distance. Implements A* search
with two modes:
  - Shortest path (minimize distance)
  - Safest path (minimize cumulative risk)

The grid cells (~500m each) become nodes, connected to their 8
neighbors. Edge costs are either Euclidean distance or the risk
score of the destination cell.
"""

import heapq
import math
import numpy as np
import pandas as pd
from collections import defaultdict


# ── Haversine distance (meters) ───────────────────────────

def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance between two lat/lon points in meters."""
    R = 6371000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── Risk Grid Graph ───────────────────────────────────────

class RiskGrid:
    """
    Graph built from collision risk data for pathfinding.

    Each grid cell is a node. Edges connect to 8 neighbors
    (N, S, E, W, NE, NW, SE, SW). Cells with no collision
    data get a default low risk (they're likely safe residential
    areas with no recorded KSI incidents). Cells over Lake Ontario
    are marked as impassable.
    """

    # Toronto shoreline approximation (lon -> minimum land latitude)
    # Anything south of this line at a given longitude is water.
    # Built from the southernmost collision records in the KSI data.
    SHORELINE = [
        (-79.65, 43.58),
        (-79.55, 43.585),
        (-79.50, 43.60),
        (-79.45, 43.62),
        (-79.40, 43.63),
        (-79.38, 43.635),
        (-79.35, 43.64),
        (-79.30, 43.66),
        (-79.25, 43.69),
        (-79.20, 43.73),
        (-79.15, 43.74),
        (-79.10, 43.75),
    ]

    def __init__(self, grid_csv_path, grid_size=0.005):
        self.grid_size = grid_size
        self.cells = {}         # (lat, lon) -> risk score
        self.cell_data = {}     # (lat, lon) -> full data dict
        self.default_risk = 0.05  # Low risk for cells with no data

        # Load risk data
        df = pd.read_csv(grid_csv_path)
        for _, row in df.iterrows():
            key = (round(row['glat'], 4), round(row['glon'], 4))
            self.cells[key] = row['route_risk']
            self.cell_data[key] = {
                'risk': row['route_risk'],
                'count': int(row['count']),
                'fatals': int(row['fatals']),
            }

        # Build set of routable cells: any cell that has data OR
        # is within 2 grid steps of a cell with data (allows routing
        # through quiet residential areas but blocks water/empty land)
        self.routable = set(self.cells.keys())
        for key in list(self.cells.keys()):
            lat, lon = key
            for dlat in range(-2, 3):
                for dlon in range(-2, 3):
                    neighbor = (round(lat + dlat * grid_size, 4),
                                round(lon + dlon * grid_size, 4))
                    if not self._is_water(neighbor[0], neighbor[1]):
                        self.routable.add(neighbor)

        # Compute bounding box
        lats = [k[0] for k in self.cells]
        lons = [k[1] for k in self.cells]
        self.lat_min = min(lats) - grid_size * 5
        self.lat_max = max(lats) + grid_size * 5
        self.lon_min = min(lons) - grid_size * 5
        self.lon_max = max(lons) + grid_size * 5

        print(f"  RiskGrid: {len(self.cells)} data cells | {len(self.routable)} routable cells")
        print(f"  Bounds: ({self.lat_min:.3f},{self.lon_min:.3f}) to ({self.lat_max:.3f},{self.lon_max:.3f})")

    def _is_water(self, lat, lon):
        """
        Check if a coordinate is over Lake Ontario using a
        piecewise-linear shoreline approximation. Returns True
        if the point is south of the shoreline (i.e. in the lake).
        """
        shore = self.SHORELINE
        # Clamp to shoreline range
        if lon <= shore[0][0]:
            return lat < shore[0][1]
        if lon >= shore[-1][0]:
            return lat < shore[-1][1]
        # Interpolate between shoreline points
        for i in range(len(shore) - 1):
            if shore[i][0] <= lon <= shore[i + 1][0]:
                frac = (lon - shore[i][0]) / (shore[i + 1][0] - shore[i][0])
                shore_lat = shore[i][1] + frac * (shore[i + 1][1] - shore[i][1])
                return lat < shore_lat
        return False

    def snap_to_grid(self, lat, lon):
        """Snap a coordinate to the nearest grid point."""
        glat = round(round(lat / self.grid_size) * self.grid_size, 4)
        glon = round(round(lon / self.grid_size) * self.grid_size, 4)
        return (glat, glon)

    def get_risk(self, node):
        """Get risk for a grid cell. Default low risk if no data."""
        return self.cells.get(node, self.default_risk)

    def is_routable(self, node):
        """Check if a cell is on land (has data nearby)."""
        return node in self.routable

    def in_bounds(self, node):
        """Check if a node is within Toronto bounds."""
        return (self.lat_min <= node[0] <= self.lat_max and
                self.lon_min <= node[1] <= self.lon_max)

    def neighbors(self, node):
        """Return 8-connected neighbors that are routable (on land)."""
        lat, lon = node
        gs = self.grid_size
        dirs = [
            (gs, 0), (-gs, 0), (0, gs), (0, -gs),   # N, S, E, W
            (gs, gs), (gs, -gs), (-gs, gs), (-gs, -gs)  # diagonals
        ]
        result = []
        for dlat, dlon in dirs:
            nb = (round(lat + dlat, 4), round(lon + dlon, 4))
            if self.in_bounds(nb) and self.is_routable(nb):
                result.append(nb)
        return result


# ── A* Search ─────────────────────────────────────────────

def heuristic_distance(a, b):
    """Haversine distance heuristic for A* (in km)."""
    return haversine(a[0], a[1], b[0], b[1]) / 1000.0


def heuristic_risk(a, b, grid):
    """
    Risk-based heuristic: estimate remaining risk as
    (straight-line distance in cells) * minimum possible risk.
    This is admissible since actual risk >= min_risk per cell.
    """
    cells_remaining = heuristic_distance(a, b) / (grid.grid_size * 111)  # approx km per degree
    return cells_remaining * grid.default_risk * 0.5  # admissible lower bound


def astar(grid, start, goal, mode="safest"):
    """
    A* pathfinding on the risk grid.

    Args:
        grid:  RiskGrid instance
        start: (lat, lon) tuple
        goal:  (lat, lon) tuple
        mode:  "safest" (minimize risk) or "shortest" (minimize distance)

    Returns:
        path:       List of (lat, lon) nodes
        total_cost: Total path cost (risk or distance)
        risk_sum:   Total accumulated risk along the path
        distance:   Total distance in km
        explored:   Number of nodes explored (for analysis)
    """
    start = grid.snap_to_grid(*start)
    goal = grid.snap_to_grid(*goal)

    # Priority queue: (cost, tiebreaker, node)
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
            # Reconstruct path
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
        if explored > 50000:  # Safety limit
            break

        for nb in grid.neighbors(current):
            # Distance between cells
            d = haversine(current[0], current[1], nb[0], nb[1]) / 1000.0

            if mode == "safest":
                # Cost = risk of destination cell (weighted by distance for diagonals)
                risk = grid.get_risk(nb)
                step_cost = risk * d
            else:
                # Cost = pure distance
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

    # No path found - return None
    return None


# ── Route Comparison ──────────────────────────────────────

def compare_routes(grid, start, goal):
    """
    Find both the shortest and safest routes and compare them.

    Returns a dict with both routes and comparison statistics.
    """
    print(f"\n  Routing from ({start[0]:.4f}, {start[1]:.4f}) to ({goal[0]:.4f}, {goal[1]:.4f})")

    shortest = astar(grid, start, goal, mode="shortest")
    safest = astar(grid, start, goal, mode="safest")

    if not shortest or not safest:
        print("  ERROR: Could not find a path!")
        return None

    # Calculate risk along the shortest path for comparison
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


# ── Main entry for testing ────────────────────────────────

if __name__ == "__main__":
    grid = RiskGrid("outputs/risk_grid.csv")

    # Test routes across Toronto
    routes = [
        ("Downtown to Scarborough",
         (43.6550, -79.3830),  # King & Yonge
         (43.7730, -79.2580)), # Scarborough Town Centre
        ("Etobicoke to East York",
         (43.6440, -79.5100),  # Kipling
         (43.6920, -79.3270)), # Danforth & Woodbine
        ("North York to Waterfront",
         (43.7670, -79.4110),  # Yonge & Sheppard
         (43.6390, -79.3810)), # Union Station area
    ]

    for name, start, goal in routes:
        print(f"\n{'='*50}")
        print(f"  {name}")
        result = compare_routes(grid, start, goal)