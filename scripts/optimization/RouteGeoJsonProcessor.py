import geojson
import polyline
import json
import geopandas as gpd
import math
from shapely.ops import unary_union
from collections import OrderedDict
import os

class RouteGeoJSONProcessor:
    """
    Processes a GeoJSON file containing routes with encoded polylines,
    and generates a new GeoJSON file with decoded polylines as LineStrings.
    """

    @classmethod
    def decode_polylines_for_routes_to_geojson(
        cls,
        pareto_json_path,
        od_matrix_geojson_path,
        output_dir,
        nodes_geojson_path
    ):
        
        # Load nodes GeoJSON and build a lookup: osmid -> concurrenceTotal and geometry
        nodes_gdf = gpd.read_file(nodes_geojson_path)
        node_lookup = {
            int(row['osmid']): {
                "concurrence": int(row['concurrenceTotal']),
                "geometry": row.geometry
            } for _, row in nodes_gdf.iterrows()
        }

        # Load all solutions
        with open(pareto_json_path, "r") as f:
            solutions = json.load(f)

        # Load the OD matrix GeoJSON
        with open(od_matrix_geojson_path, "r") as f:
            od_data = geojson.load(f)

        # Build a lookup for (origin, dest) -> feature
        od_lookup = {}
        for feature in od_data["features"]:
            props = feature["properties"]
            origin = props.get("origin_id")
            dest = props.get("destination_id")
            if origin is not None and dest is not None:
                od_lookup[(int(origin), int(dest))] = feature

        # For each solution
        for solution_index, solution in enumerate(solutions):
            features = []
            for r, od_id_list in enumerate(solution["od_routes"]):
                route_coords = []
                total_distance = 0
                total_time = 0
                visited_nodes = set()

                for od_id in od_id_list:
                    try:
                        origin, dest = map(int, od_id.split('-'))
                    except Exception:
                        print(f"Could not parse od_id: {od_id}")
                        continue

                    feature = od_lookup.get((origin, dest))
                    if feature:
                        props = feature.get("properties", {})
                        polyline_str = props.get("polyline")
                        if polyline_str:
                            try:
                                decoded_coords = polyline.decode(polyline_str)
                                # Convert to (lon, lat) tuples for geojson LineString
                                decoded_coords = [(lon, lat) for lat, lon in decoded_coords]
                                if route_coords and decoded_coords:
                                    route_coords.extend(decoded_coords[1:])
                                else:
                                    route_coords.extend(decoded_coords)
                            except Exception as e:
                                print(f"Failed to decode polyline for {origin}->{dest}: {e}")
                                continue

                        total_distance += float(props.get("distance_meters", 0))
                        total_time += float(props.get("duration_seconds", 0))
                        visited_nodes.add(origin)
                        visited_nodes.add(dest)

                # --- Compute buffer-based area using 200m radius per node ---
                node_buffers = [node_lookup[n]['geometry'].buffer(200) for n in visited_nodes if n in node_lookup]
                area_geom = unary_union(node_buffers)
                total_area = area_geom.area if not area_geom.is_empty else 0

                # --- Total concurrence ---
                total_conc = sum(node_lookup[n]["concurrence"] for n in visited_nodes if n in node_lookup)

                # Offset amount in meters — change this if needed
                offset_meters = (r - len(solution["od_routes"]) // 2) * 0.5  # center around 0

                # Apply offset
                route_coords = RouteGeoJSONProcessor.offset_coords(route_coords, offset_meters)
                
                if route_coords:
                    feature = OrderedDict()
                    feature["type"] = "Feature"
                    feature["properties"] = {
                        "route_index": r,
                        "total_distance_meters": total_distance,
                        "total_duration_seconds": total_time,
                        "concurrence_served": total_conc,
                        "total_area": total_area
                    }
                    feature["geometry"] = {
                        "type": "LineString",
                        "coordinates": route_coords
                    }
                    features.append(feature)

            fc = OrderedDict()
            fc["type"] = "FeatureCollection"
            fc["features"] = features

            output_geojson_path = os.path.join(
                output_dir, f"solution{solution_index}routes_decoded.geojson"
            )
            with open(output_geojson_path, "w") as f:
                geojson.dump(fc, f)
            print(f"Decoded routes for solution {solution_index} saved to {output_geojson_path}")
            
    def offset_coords(coords, offset_meters):
        """
        Apply a small perpendicular offset to each segment of the route so the don't overlap
        """
        offset_coords = []
        for i in range(len(coords)):
            if i == 0:
                # Estimate direction from the next point
                dx = coords[i + 1][0] - coords[i][0]
                dy = coords[i + 1][1] - coords[i][1]
            elif i == len(coords) - 1:
                # Estimate direction from the previous point
                dx = coords[i][0] - coords[i - 1][0]
                dy = coords[i][1] - coords[i - 1][1]
            else:
                dx = coords[i + 1][0] - coords[i - 1][0]
                dy = coords[i + 1][1] - coords[i - 1][1]

            # Normalize the direction vector
            length = math.hypot(dx, dy)
            if length == 0:
                offset_coords.append(coords[i])
                continue
            dx /= length
            dy /= length

            # Perpendicular vector (right-hand rule)
            perp_x = -dy
            perp_y = dx

            # Offset in degrees
            offset_deg_lat = (offset_meters / 111000) * perp_y
            offset_deg_lon = (offset_meters / (111000 * math.cos(math.radians(coords[i][1])))) * perp_x

            new_x = coords[i][0] + offset_deg_lon
            new_y = coords[i][1] + offset_deg_lat
            offset_coords.append((new_x, new_y))

        return offset_coords

RouteGeoJSONProcessor.decode_polylines_for_routes_to_geojson(
    "data_files/result_files/pareto_solutions_13_routes.json",
    "data_files/base_files/origin_destinations_matrix_6.geojson",
    "data_files/solutions/n13RoutesSolution",
    "data_files/base_files/nodes6_with_200m_concurrence_assigned_centered.geojson"
)

RouteGeoJSONProcessor.decode_polylines_for_routes_to_geojson(
    "data_files/result_files/initial_population_13_routes.json",
    "data_files/base_files/origin_destinations_matrix_6.geojson",
    "data_files/initial_population",
    "data_files/base_files/nodes6_with_200m_concurrence_assigned_centered.geojson"
)