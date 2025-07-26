import geopandas as gpd
import osmnx as ox
from tqdm import tqdm

class NodeUpdater:
    """
    Updates nodes with area-weighted population within a specified buffer radius.
    All file paths are relative to the data_files directory.
    """

    def __init__(self, nodes_filename, grid_filename, output_filename, buffer_radius=200, utm_crs="EPSG:32719"):
        """
        Assigns population to nodes according to overlapping radius with population grid cells
        """
        self.nodes_path = nodes_filename
        self.grid_path = grid_filename
        self.output_path = output_filename
        self.buffer_radius = buffer_radius
        self.utm_crs = utm_crs
        self.nodes_gdf = None
        self.grid_gdf = None

    def load_data(self):
        """Load nodes and grid GeoDataFrames and reproject to metric CRS."""
        self.nodes_gdf = gpd.read_file(self.nodes_path)
        self.grid_gdf = gpd.read_file(self.grid_path)
        self.nodes_gdf = self.nodes_gdf.to_crs(self.utm_crs)
        self.grid_gdf = self.grid_gdf.to_crs(self.utm_crs)

    def assign_area_weighted_population(self):
        """Assign area-weighted population within buffer to each node."""
        grid_sindex = self.grid_gdf.sindex
        assigned_populations = []
        tqdm.pandas(desc="Processing nodes")
        
        for node_geom in tqdm(self.nodes_gdf.geometry, desc="Calculating area-weighted population"):
            buffer = node_geom.buffer(self.buffer_radius)
            # Get overlapping grid cells
            possible_matches_index = list(grid_sindex.intersection(buffer.bounds))
            possible_matches = self.grid_gdf.iloc[possible_matches_index]
            precise_matches = possible_matches[possible_matches.intersects(buffer)]
            weighted_population = 0.0
            for _, cell in precise_matches.iterrows():
                intersection = buffer.intersection(cell.geometry)
                if not intersection.is_empty:
                    weight = intersection.area / cell.geometry.area
                    weighted_population += weight * cell['population']
            assigned_populations.append(weighted_population)
        self.nodes_gdf['assigned_pop'] = assigned_populations

    def reproject_to_wgs84(self):
        """Reproject nodes to WGS84."""
        self.nodes_gdf = self.nodes_gdf.to_crs("EPSG:4326")

    def drop_population_column(self):
        """
        Drop the 'population' column from the nodes GeoDataFrame if it exists.
        """
        if "population" in self.nodes_gdf.columns:
            self.nodes_gdf = self.nodes_gdf.drop(columns=["population"])
            print("Dropped 'population' column.")
        else:
            print("'population' column not found.")

    def reorder_assigned_pop_column(self):
        """
        Reorder columns so 'assigned_pop' is in the 4th position.
        """
        if "assigned_pop" not in self.nodes_gdf.columns:
            raise ValueError("'assigned_pop' column not found in the GeoDataFrame.")

        cols = list(self.nodes_gdf.columns)
        cols.remove("assigned_pop")
        cols.insert(3, "assigned_pop")  # 4th position (index 3)
        self.nodes_gdf = self.nodes_gdf[cols]

    def round_assigned_pop_and_osmid(self):
        """
        Round 'assigned_pop' and 'osmid' columns to integers if they exist.
        """
        if "assigned_pop" in self.nodes_gdf.columns:
            self.nodes_gdf["assigned_pop"] = self.nodes_gdf["assigned_pop"].round().astype("Int64")
        if "osmid" in self.nodes_gdf.columns:
            self.nodes_gdf["osmid"] = self.nodes_gdf["osmid"].round().astype("Int64")
            
    def save(self):
        """Save the GeoDataFrame as GeoJSON."""
        self.nodes_gdf.to_file(self.output_path, driver="GeoJSON")
        print(f"Saved nodes with area-weighted population (WGS84) to {self.output_path}")

    def run(self):
        """Run the full update pipeline."""
        self.load_data()
        self.assign_area_weighted_population()
        self.reproject_to_wgs84()
        self.drop_population_column()
        self.reorder_assigned_pop_column()
        self.round_assigned_pop_and_osmid()
        self.save()

    @staticmethod
    def snap_nodes_to_intersection_centers(input_geojson_path: str, output_geojson_path: str):
        print(f"Loading input nodes: {input_geojson_path}")
        nodes = gpd.read_file(input_geojson_path)

        if 'street_count' not in nodes.columns:
            raise ValueError("'street_count' column is required in input GeoJSON")

        # Project nodes to metric CRS for accurate distance calculations
        nodes = nodes.to_crs(epsg=3857)

        print("Downloading street network from OSM...")
        G = ox.graph_from_place("Cercado, Cochabamba, Bolivia", network_type='drive')

        # Project graph to metric CRS (EPSG:3857) for spatial ops
        G_proj = ox.project_graph(G, to_crs="EPSG:3857")

        print("Computing real intersection centers from OSM topology...")
        intersections = ox.consolidate_intersections(G_proj, tolerance=20, rebuild_graph=False)

        # intersections is a GeoDataFrame already in EPSG:3857, no need to reproject

        updated_geometries = []

        for idx, row in nodes.iterrows():
            point = row.geometry

            if row['street_count'] < 2:
                # Not an intersection, keep original point
                updated_geometries.append(point)
                continue

            # Find nearest real intersection center
            nearest_idx = intersections.geometry.distance(point).idxmin()
            nearest_center = intersections.geometry.loc[nearest_idx]
            updated_geometries.append(nearest_center)

        # Update geometry column
        nodes['geometry'] = updated_geometries

        # Reproject back to WGS84 for saving
        nodes = nodes.to_crs(epsg=4326)

        # Update x and y columns
        nodes['x'] = nodes.geometry.x
        nodes['y'] = nodes.geometry.y

        print(f"Saving updated nodes to: {output_geojson_path}")
        nodes.to_file(output_geojson_path, driver="GeoJSON")