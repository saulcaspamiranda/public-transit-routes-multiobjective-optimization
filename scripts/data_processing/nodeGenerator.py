import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
        
class NodeGenerator:
    """
    Generates nodes along major routes within populated grid cells,
    ensuring nodes are at least a specified distance apart.
    """

    def __init__(self, place_name, grid_path, min_distance, restriction_area_path, utm_crs="EPSG:32719"):
        self.place_name = place_name
        self.grid_path = grid_path
        self.min_distance = min_distance
        self.restriction_area_path = restriction_area_path if restriction_area_path else None
        self.utm_crs = utm_crs
        self.nodes = None
        self.edges = None
        self.major_nodes = None
        self.grid_gdf = None
        self.nodes_in_grids = None
        self.filtered_nodes = None

    def load_road_network(self):
        """Load the road network and extract nodes and edges."""
        G = ox.graph_from_place(self.place_name, network_type="drive", simplify=False)
        self.nodes, self.edges = ox.graph_to_gdfs(G, nodes=True, edges=True)

    def filter_major_roads(self):
        """Filter edges to keep only major road types."""
        major_road_types = [
            'motorway', 'trunk', 'primary', 'secondary',
            'tertiary', 'motorway_link', 'primary_link', 'secondary_link'
        ]
        def is_major(hw):
            if isinstance(hw, list):
                return any(h in major_road_types for h in hw)
            return hw in major_road_types
        major_edges = self.edges[self.edges['highway'].apply(is_major)]
        major_node_ids = set(major_edges.index.get_level_values('u')) | set(major_edges.index.get_level_values('v'))
        self.major_nodes = self.nodes[self.nodes.index.isin(major_node_ids)]

    def load_and_filter_grid(self):
        """Load the population grid and filter to populated cells."""
        self.grid_gdf = gpd.read_file(self.grid_path)
        self.grid_gdf = self.grid_gdf[self.grid_gdf['population'] > 0]

    def reproject_data(self):
        """Reproject nodes and grid to the same metric CRS."""
        self.major_nodes = self.major_nodes.to_crs(self.utm_crs)
        self.grid_gdf = self.grid_gdf.to_crs(self.utm_crs)

    def filter_nodes_in_grids(self):
        """Keep only nodes that fall within populated grid cells."""
        nodes_in_grids = gpd.sjoin(self.major_nodes, self.grid_gdf, how='inner', predicate='within')
        self.nodes_in_grids = nodes_in_grids.reset_index().rename(columns={"index": "node_id"})

    def greedy_filter_nodes(self):
        """Greedy filtering to keep nodes at least min_distance apart."""
        coords = np.array([(geom.x, geom.y) for geom in self.nodes_in_grids.geometry])
        tree = BallTree(coords, metric='euclidean')
        selected_indices = []
        used = np.full(len(coords), False)
        for i in range(len(coords)):
            if not used[i]:
                selected_indices.append(i)
                inds = tree.query_radius([coords[i]], r=self.min_distance)[0]
                used[inds] = True
        self.filtered_nodes = self.nodes_in_grids.iloc[selected_indices]

    def add_nodes_in_restriction_area(self):
        """
        Add street intersection nodes within the restriction area, prioritizing those on major roads.
        """

        # Load restriction area polygon
        restriction_area = gpd.read_file(self.restriction_area_path)
        if restriction_area.crs != self.utm_crs:
            restriction_area = restriction_area.to_crs(self.utm_crs)

        # Define major road types
        major_road_types = [
            'motorway', 'trunk', 'primary', 'secondary',
            'tertiary', 'motorway_link', 'primary_link', 'secondary_link'
        ]
        def is_major(hw):
            if isinstance(hw, list):
                return any(h in major_road_types for h in hw)
            return hw in major_road_types

        # Filter edges to keep only major road types
        major_edges = self.edges[self.edges['highway'].apply(is_major)]
        major_node_ids = set(major_edges.index.get_level_values('u')) | set(major_edges.index.get_level_values('v'))

        # Find intersection nodes (degree >= 3) that are on major roads
        node_degree = self.edges.groupby('u').size().add(self.edges.groupby('v').size(), fill_value=0)
        intersection_ids = node_degree[node_degree >= 3].index
        # Only keep intersection nodes that are also on major roads
        major_intersection_ids = [nid for nid in intersection_ids if nid in major_node_ids]
        intersection_nodes = self.nodes.loc[self.nodes.index.isin(major_intersection_ids)].copy()
        intersection_nodes = intersection_nodes.to_crs(self.utm_crs)

        # Keep only intersections within the restriction area
        intersections_in_area = intersection_nodes[intersection_nodes.within(restriction_area.unary_union)].copy()

        # Greedy filter intersections to be at least 160m apart (diagonally distributed)
        coords = np.array([(geom.x, geom.y) for geom in intersections_in_area.geometry])
        selected_indices = []
        used = np.full(len(coords), False)
        for i in range(len(coords)):
            if not used[i]:
                selected_indices.append(intersections_in_area.index[i])
                inds = BallTree(coords, metric='euclidean').query_radius([coords[i]], r=150)[0]
                used[inds] = True
        dense_nodes = intersections_in_area.loc[selected_indices]

        # Separate previously filtered nodes inside and outside the restriction area
        mask_within = self.filtered_nodes.within(restriction_area.unary_union)
        filtered_inside = self.filtered_nodes[mask_within].copy()
        filtered_outside = self.filtered_nodes[~mask_within].copy()

        # Ensure all GeoDataFrames are in the same CRS
        filtered_outside = filtered_outside.to_crs(self.utm_crs)
        filtered_inside = filtered_inside.to_crs(self.utm_crs)

        # Combine: outside nodes + new dense intersection nodes inside restriction area + previous nodes inside restriction area, drop duplicates by geometry
        combined = pd.concat([filtered_outside, dense_nodes, filtered_inside]).drop_duplicates(subset='geometry')
        self.filtered_nodes = combined.reset_index(drop=True)
    
    def save_outputs(self, out_dir):
        """Save intermediate and filtered outputs."""
        filtered_nodes_wgs84_path = out_dir
        filtered_nodes_wgs84 = self.filtered_nodes.to_crs("EPSG:4326")
        filtered_nodes_wgs84.to_file(filtered_nodes_wgs84_path, driver="GeoJSON")

        print(f"Original candidate nodes: {len(self.nodes_in_grids)}")
        print(f"Filtered nodes (greater than {self.min_distance}m apart): {len(self.filtered_nodes)}")

    def run(self, out_dir):
        """Run the full node generation pipeline."""
        self.load_road_network()
        self.filter_major_roads()
        self.load_and_filter_grid()
        self.reproject_data()
        self.filter_nodes_in_grids()
        self.greedy_filter_nodes()
        self.save_outputs(out_dir)

    def run_with_restriction_area(self, out_dir):
        """Run the full node generation pipeline."""
        self.load_road_network()
        self.filter_major_roads()
        self.load_and_filter_grid()
        self.reproject_data()
        self.filter_nodes_in_grids()
        self.greedy_filter_nodes()
        self.add_nodes_in_restriction_area()
        self.save_outputs(out_dir)
