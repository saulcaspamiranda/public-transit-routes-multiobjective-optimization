import json
import geopandas as gpd
from shapely.geometry import shape

class DataLoader:
    def __init__(self):
        self.nodes_gdf = None
        self.starting_nodes_gdf = None
        self.population_grid = None
        self.od_df = None
        self.meters_crs = "EPSG:32719"
        self.latitude_longitude_crs = "EPSG:4326"  # CRS for latitude/longitude coordinates

    def load_nodes(self, nodes_path):
        print(f"Loading nodes from: {nodes_path}")
        try:
            gdf = gpd.read_file(nodes_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load GeoJSON: {e}")

        if 'osmid' not in gdf.columns and gdf.index.name != 'osmid':
            raise ValueError("'osmid' column not found in nodes file.")

        if gdf.index.name != 'osmid':
            gdf = gdf.set_index('osmid')

        if 'assigned_pop' not in gdf.columns:
            raise ValueError("Missing 'assigned_pop' column for population data.")

        if gdf.geometry.is_empty.any():
            raise ValueError("Some geometries are empty or missing in the nodes GeoJSON.")
        
        print(f"Loaded {len(gdf)} nodes with geometry and population.")
        self.nodes_gdf = gdf
        return gdf

    def load_od_matrix(self, od_matrix_path):
        reproject=True
        print(f"Loading OD matrix from: {od_matrix_path}")
        
        with open(od_matrix_path, 'r', encoding='utf-8') as f:
            geojson_data = json.load(f)

        if geojson_data['type'] != 'FeatureCollection':
            raise ValueError("Expected a GeoJSON FeatureCollection.")

        records = []
        geometries = []
        for feature in geojson_data['features']:
            props = feature['properties']
            geometries.append(shape(feature['geometry']))
            records.append(props)

        gdf = gpd.GeoDataFrame(records, geometry=geometries, crs=self.latitude_longitude_crs)

        if 'origin_id' not in gdf.columns or 'destination_id' not in gdf.columns:
            raise ValueError("OD GeoJSON must have 'origin_id' and 'destination_id' in properties.")
        
        if reproject:
            gdf = gdf.to_crs(self.meters_crs)

        gdf = gdf.set_index(['origin_id', 'destination_id'])
        self.od_df = gdf
        print(f"Loaded {len(gdf)} OD pairs with geometry (reprojected: {reproject}).")
        return gdf

    def load_starting_node_pairs(self, starting_node_pairs_path):
        path = starting_node_pairs_path
        print(f"Loading starting node pairs from: {path}")
        try:
            gdf = gpd.read_file(path)
        except Exception as e:
            raise RuntimeError(f"Failed to load starting node pairs GeoJSON: {e}")

        required_cols = ['origin_id', 'destination_id', 'apart_distance']
        for col in required_cols:
            if col not in gdf.columns:
                raise ValueError(f"Missing required column '{col}' in starting node pairs GeoJSON.")

        if gdf.geometry.is_empty.any():
            raise ValueError("Some geometries in starting node pairs are empty or invalid.")

        print(f"Loaded {len(gdf)} starting node pairs.")
        gdf = gdf.to_crs(self.meters_crs)
        self.starting_nodes_gdf = gdf
        return gdf
            
    def load_orbital_stop_quads(self, orbital_stop_quads_path):
        print(f"Loading orbital stop quads from: {orbital_stop_quads_path}")
        
        try:
            gdf = gpd.read_file(orbital_stop_quads_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load orbital stop quads: {e}")

        if 'geometry' not in gdf.columns or gdf.geometry.is_empty.any():
            raise ValueError("Orbital stop quads must have valid geometries.")

        gdf["osmid"] = gdf["osmid_str"].apply(lambda s: list(map(int, s.split(","))))
        gdf = gdf.to_crs(self.meters_crs)
        print(f"Loaded {len(gdf)} orbital stop quads.")
        
        return gdf
    
    def load_population_grid(self, population_grid_path):
        full_path = population_grid_path
        print(f"Loading population grid from: {full_path}")
        
        try:
            gdf = gpd.read_file(full_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load population grid: {e}")
        
        required_columns = ['FID', 'population', 'geometry']
        for col in required_columns:
            if col not in gdf.columns:
                raise ValueError(f"Missing required column '{col}' in population grid.")

        if not all(gdf.geometry.type == 'Polygon'):
            raise ValueError("Population grid must contain only Polygon geometries.")
        
        gdf = gdf.to_crs(self.meters_crs)

        print(f"Loaded population grid with {len(gdf)} cells.")
        return gdf