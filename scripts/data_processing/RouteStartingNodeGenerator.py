import geopandas as gpd
import numpy as np
import random
from shapely.geometry import MultiPoint
from shapely.geometry import Point, LineString
from shapely.affinity import rotate, scale, translate

class RouteStartingNodeGenerator:
    def __init__(self, node_geojson_path, cochabamba_boundaries_path, restriction_area_path, pairs_output_geojson_path,
                 quads_output_geojson_path, min_pair_route_distance_km, min_orbital_route_distance_km,
                 max_orbital_route_distance_km, distance_from_border, border_top_concurrence_percentile,
                 restricted_area_top_concurrence_percentile_nodes, orbital_vertices_top_concurrence_percentile_nodes):
        self.node_geojson_path = node_geojson_path
        self.cochabamba_boundaries_path = cochabamba_boundaries_path
        self.restriction_area_path = restriction_area_path
        self.pairs_output_geojson_path = pairs_output_geojson_path
        self.quads_output_geojson_path = quads_output_geojson_path
        self.min_pair_distance_km = min_pair_route_distance_km
        self.min_orbital_route_distance_km= min_orbital_route_distance_km
        self.max_orbital_route_distance_km = max_orbital_route_distance_km
        self.distance_from_border = distance_from_border
        self.border_top_concurrence_percentile = border_top_concurrence_percentile
        self.restricted_area_top_concurrence_percentile_nodes = restricted_area_top_concurrence_percentile_nodes
        self.orbital_vertices_top_concurrence_percentile_nodes = orbital_vertices_top_concurrence_percentile_nodes
        self.projected_crs = "EPSG:32719"
        self.geographic_crs = "EPSG:4326"
        self.nodes_gdf = None
        self.restriction_area = None
        self.cochabamba_boundaries = None
        self.seen_quads = []

    def load_data(self):
        self.nodes_gdf = gpd.read_file(self.node_geojson_path).to_crs(self.projected_crs)
        self.restriction_area = gpd.read_file(self.restriction_area_path).to_crs(self.projected_crs)
        self.cochabamba_boundaries = gpd.read_file(self.cochabamba_boundaries_path).to_crs(self.projected_crs)
        print("Nodes CRS:", self.nodes_gdf.crs)
        print("Restriction area CRS:", self.restriction_area.crs if hasattr(self.restriction_area, 'crs') else 'Not a GeoDataFrame')
    
    def keep_border_and_near_border_nodes(self, ensured_nodes_gdf):
        ensured_proj = ensured_nodes_gdf.to_crs(self.projected_crs)

        # Create convex hull polygon and get its boundary (a LineString)
        hull = ensured_proj.union_all().convex_hull
        border_line = hull.boundary

        # Buffer only the boundary line (not the entire polygon)
        border_buffer = border_line.buffer(self.distance_from_border)

        # Keep nodes that are either on the border or within 1km of it
        kept_nodes = ensured_proj[ensured_proj.geometry.intersects(border_buffer)]
        print(f"Kept {len(kept_nodes)} nodes near or at the border from convex hull edge).")

        return kept_nodes.to_crs(self.geographic_crs)

    def generate_node_pairs(self, best_gdf):
        best_proj = best_gdf.to_crs(self.projected_crs)
        pairs = []
        meters_per_kilometer = 1000
        for i, row1 in best_proj.iterrows():
            for j, row2 in best_proj.iterrows():
                if i < j:
                    dist_m = row1.geometry.distance(row2.geometry)
                    if dist_m >= self.min_pair_distance_km * meters_per_kilometer:
                        point_pair = MultiPoint([row1.geometry, row2.geometry])
                        pairs.append({
                            'origin_id': row1['osmid'],
                            'destination_id': row2['osmid'],
                            'apart_distance': round(dist_m),
                            'geometry': point_pair
                        })

        if pairs:
            return gpd.GeoDataFrame(pairs, geometry='geometry', crs=self.projected_crs).to_crs(self.geographic_crs)
        else:
            print("No valid node pairs generated.")
            return gpd.GeoDataFrame(
                columns=['origin_id', 'destination_id', 'apart_distance', 'geometry'],
                geometry='geometry',
                crs=self.geographic_crs
            )

    def filter_top_percentile_nodes(self, percentile):
        if 'assigned_pop' in self.nodes_gdf.columns and not self.nodes_gdf.empty:
            threshold = self.nodes_gdf['assigned_pop'].quantile(percentile)
            filtered = self.nodes_gdf[self.nodes_gdf['assigned_pop'] >= threshold]
            print(f"Filtered down to {len(filtered)} nodes out of {len(self.nodes_gdf)} based on top {int((1 - percentile) * 100)}% assigned_pop")
            return filtered
        return self.nodes_gdf

    def generate_node_quads(self, num_quads):
        results = []
        seen_quad_keys = set()
        cbba_union = self.cochabamba_boundaries.union_all()
        restriction_polygon = self.restriction_area.geometry.iloc[0]

        attempts = 0
        max_attempts = num_quads * 20

        while len(results) < num_quads and attempts < max_attempts:
            attempts += 1
            # Select nodes within the restriction polygon
            candidates = self.nodes_gdf[self.nodes_gdf.geometry.within(restriction_polygon)]
            if candidates.empty:
                print("No nodes inside restriction area.")
                break
            restriction_area_nodes_concurrence_threshold = candidates["concurrenceTotal"].quantile(self.restricted_area_top_concurrence_percentile_nodes)
            orbital_vertice_nodes_concurrence_threshold = self.nodes_gdf["concurrenceTotal"].quantile(self.orbital_vertices_top_concurrence_percentile_nodes)
        
            # Only keep nodes in the top percentage of concurrence
            top_candidates = candidates[candidates["concurrenceTotal"] >= restriction_area_nodes_concurrence_threshold]
            if top_candidates.empty:
                print("No high-concurrence nodes in restriction area.")
                continue

            # Sample one from the top 50%
            fixed_node = top_candidates.sample(1).iloc[0]
            fixed_geom = fixed_node.geometry
            fixed_angle_deg = 0

            a = random.uniform(2000, 4000)
            b = random.uniform(1500, 3500)
            angle_deg = random.uniform(0, 360)

            theta = np.radians(fixed_angle_deg)
            dx = a * np.cos(theta)
            dy = b * np.sin(theta)

            rot = np.radians(-angle_deg)
            offset_x = dx * np.cos(rot) - dy * np.sin(rot)
            offset_y = dx * np.sin(rot) + dy * np.cos(rot)

            center_x = fixed_geom.x - offset_x
            center_y = fixed_geom.y - offset_y

            ellipse = Point(0, 0).buffer(1, resolution=16)
            ellipse = scale(ellipse, a, b)
            ellipse = rotate(ellipse, angle_deg)
            ellipse = translate(ellipse, xoff=center_x, yoff=center_y)

            if not cbba_union.contains(ellipse):
                continue

            quad_nodes = [int(fixed_node["osmid"])]
            node_geoms = [fixed_geom]

            for ang in [90, 180, 270]:
                theta = np.radians(ang)
                dx = a * np.cos(theta)
                dy = b * np.sin(theta)

                offset_x = dx * np.cos(rot) - dy * np.sin(rot)
                offset_y = dx * np.sin(rot) + dy * np.cos(rot)

                px = center_x + offset_x
                py = center_y + offset_y
                vertex = Point(px, py)

                nearby = self.nodes_gdf[self.nodes_gdf.geometry.distance(vertex) <= 500]
                if nearby.empty:
                    break

                best_node = nearby.sort_values("concurrenceTotal", ascending=False).iloc[0]

                # Skip if this node is below the top % threshold
                if best_node["concurrenceTotal"] < orbital_vertice_nodes_concurrence_threshold:
                    break

                quad_nodes.append(int(best_node["osmid"]))
                node_geoms.append(best_node.geometry)

            if len(quad_nodes) == 4:
                quad_key = tuple(sorted(quad_nodes))
                if quad_key in seen_quad_keys:
                    continue
                seen_quad_keys.add(quad_key)

                # Close loop
                line = LineString(node_geoms + [node_geoms[0]])

                # Validate route length
                route_distance = line.length  # in meters (projected CRS)
                if not (self.min_orbital_route_distance_km * 1000 <= route_distance <= self.max_orbital_route_distance_km * 1000):
                    continue

                results.append({
                    "quad_id": len(results),
                    "osmid_str": ",".join(map(str, quad_nodes)),  # <- stringified list
                    "geometry": line
                })

        if not results:
            print("No valid quads generated.")
            return gpd.GeoDataFrame(columns=["quad_id", "osmid", "geometry"], geometry="geometry", crs=self.nodes_gdf.crs)

        gdf = gpd.GeoDataFrame(results, geometry="geometry", crs=self.projected_crs)
        gdf = gdf.to_crs(self.geographic_crs)
        return gdf

    def save_pairs_output(self, gdf):
        gdf.to_file(self.pairs_output_geojson_path, driver="GeoJSON")
        print(f"Saved {len(gdf)} node pairs to: {self.pairs_output_geojson_path}")

    def save_quads_output(self, gdf):
        gdf.to_file(self.quads_output_geojson_path, driver="GeoJSON")
        print(f"Saved {len(gdf)} node quads to: {self.quads_output_geojson_path}")
    
    def run(self):
        self.load_data()
        quads = self.generate_node_quads(num_quads=1500)
        top_nodes = self.filter_top_percentile_nodes(percentile=self.border_top_concurrence_percentile)
        border_nodes = self.keep_border_and_near_border_nodes(top_nodes)
        pairs = self.generate_node_pairs(border_nodes)
        self.save_quads_output(quads)
        self.save_pairs_output(pairs)

# Example usage:
RouteStartingNodeGenerator(
    node_geojson_path="data_files/base_files/nodes6_with_200m_concurrence_assigned_centered.geojson",
    cochabamba_boundaries_path="data_files/base_files/cochabamba_boundaries.geojson",
    restriction_area_path="data_files/base_files/vehicular_restriction_area.geojson",
    pairs_output_geojson_path="data_files/base_files/straight_stop_node_pairs.geojson",
    quads_output_geojson_path="data_files/base_files/orbital_stop_quads.geojson",
    min_pair_route_distance_km=8,
    min_orbital_route_distance_km=13,
    max_orbital_route_distance_km=20,
    distance_from_border=1800, # meters buffer from the border
    border_top_concurrence_percentile=0.2,
    restricted_area_top_concurrence_percentile_nodes=0.7,
    orbital_vertices_top_concurrence_percentile_nodes=0.3
).run()
