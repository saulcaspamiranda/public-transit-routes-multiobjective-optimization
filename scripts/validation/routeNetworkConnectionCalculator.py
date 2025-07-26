import geopandas as gpd
from itertools import combinations
import geopandas as gpd


class RouteNetworkConnectionCalculator:
    """
    Gets the routes from a geojson file and sum different combinations of 1km grid cells
    that their routes go through as the network connection value.
    """
    
    def __init__(self):
        self.grid_path:str
        self.routes_path:str
        self.grid_gdf:str
        self.routes_gdf:str

    def load_files(self, grid_path, routes_path):
        self.grid_path = grid_path
        self.routes_path = routes_path
        self.grid_gdf = gpd.read_file(grid_path)
        self.routes_gdf = gpd.read_file(routes_path)
        
        # Validate expected columns
        if "FID" not in self.grid_gdf.columns:
            raise ValueError("Grid GeoJSON must contain 'FID' column.")
        if "geometry" not in self.routes_gdf.columns or not all(self.routes_gdf.geometry.geom_type == "LineString"):
            raise ValueError("Routes GeoJSON must contain LineString geometries.")

        # Ensure same CRS
        if self.grid_gdf.crs != self.routes_gdf.crs:
            self.routes_gdf = self.routes_gdf.to_crs(self.grid_gdf.crs)

    def get_grid_cells_for_route(self, route_geom):
        """Returns the list of FIDs of grid cells the route intersects."""
        intersecting_cells = self.grid_gdf[self.grid_gdf.intersects(route_geom)]
        return list(intersecting_cells["FID"])

    def compute_unique_pairs_for_route(self, fids):
        """Returns all unique unordered pairs (combinations) of FIDs."""
        return set(tuple(sorted(pair)) for pair in combinations(set(fids), 2))

    def get_network_connection_sum(self):
        """Main method to compute unique node pairs connections covered by all routes."""
        all_pairs = set()

        for _, route in self.routes_gdf.iterrows():
            route_fids = self.get_grid_cells_for_route(route.geometry)
            route_pairs = self.compute_unique_pairs_for_route(route_fids)
            all_pairs.update(route_pairs)

        print(f"Total node connections in {self.routes_path}:\n {len(all_pairs)} connections")
        return len(all_pairs)

analyzer = RouteNetworkConnectionCalculator()

analyzer.load_files("data_files/base_files/cochabamba_1000m_population_grid_epsg4326.geojson",
                    "data_files/solutions/n13RoutesSolution/solution34routes_decoded.geojson")
total_pairs = analyzer.get_network_connection_sum()

analyzer.load_files("data_files/base_files/cochabamba_1000m_population_grid_epsg4326.geojson",
                    "data_files/solutions/n13RoutesSolution/solution198routes_decoded.geojson")
total_pairs = analyzer.get_network_connection_sum()

analyzer.load_files("data_files/base_files/cochabamba_1000m_population_grid_epsg4326.geojson",
                    "data_files/solutions/n13RoutesSolution/solution30routes_decoded.geojson")
total_pairs = analyzer.get_network_connection_sum()