import time
import json
import geopandas as gpd
from haversine import haversine, Unit
import googlemaps

class OriginDestinationMatrixGenerator:
    """
    Generates an OD matrix with distances and travel times for node pairs within a distance threshold,
    using the Google Directions API.
    """

    def __init__(
        self,
        input_filename,
        output_filename,
        api_key,
        max_qps=10,
        max_retries=3,
        distance_threshold_km=1.3
    ):
        self.input_path = input_filename
        self.output_path = output_filename
        self.api_key = api_key
        self.max_qps = max_qps
        self.request_interval = 1.0 / max_qps
        self.max_retries = max_retries
        self.distance_threshold_km = distance_threshold_km
        self.gmaps = googlemaps.Client(key=self.api_key)
        self.nodes = None
        self.od_pairs = []
        self.results = []
        print(f"distance_threshold_km = {self.distance_threshold_km}")

    def load_nodes(self):
        """Load nodes from GeoJSON and prepare for OD pair generation."""
        gdf = gpd.read_file(self.input_path)
        gdf = gdf.to_crs("EPSG:4326")
        nodes = gdf[["geometry", "osmid"]].copy()
        nodes = nodes.rename(columns={"osmid": "node_id"})
        nodes["lat"] = nodes.geometry.y
        nodes["lon"] = nodes.geometry.x
        self.nodes = nodes

    def generate_od_pairs(self):
        """Generate OD pairs within the distance threshold."""
        coords = list(zip(self.nodes["lat"], self.nodes["lon"], self.nodes["node_id"]))
        for i, (lat1, lon1, id1) in enumerate(coords):
            for j, (lat2, lon2, id2) in enumerate(coords):
                if id1 != id2:
                    dist = haversine((lat1, lon1), (lat2, lon2), unit=Unit.KILOMETERS)
                    if dist <= self.distance_threshold_km:
                        self.od_pairs.append((id1, (lat1, lon1), id2, (lat2, lon2)))
        print(f"Total OD pairs within {self.distance_threshold_km} km: {len(self.od_pairs)}")

    def get_directions_with_retry(self, origin, destination):
        """Query Google Directions API with retries and exponential backoff."""
        for attempt in range(self.max_retries):
            try:
                result = self.gmaps.directions(origin, destination, mode="driving")
                if result:
                    leg = result[0]['legs'][0]
                    overview_polyline = result[0].get("overview_polyline", {}).get("points", "")
                    return {
                        "distance_meters": leg['distance']['value'],
                        "duration_seconds": leg['duration']['value'],
                        "polyline": overview_polyline
                    }
                else:
                    return None
            except Exception as e:
                print(f"Error (attempt {attempt + 1}): {e}")
                time.sleep(2 ** attempt)  # exponential backoff
        return None

    def request_od_metrics(self):
        """Request distances and travel times for all OD pairs with throttling."""
        for i, (id1, coord1, id2, coord2) in enumerate(self.od_pairs):
            print(f"Request {i + 1}/{len(self.od_pairs)}: {id1} to {id2}")
            metrics = self.get_directions_with_retry(coord1, coord2)
            if metrics:
                self.results.append({
                    "origin_id": id1,
                    "destination_id": id2,
                    **metrics
                })
            time.sleep(self.request_interval)

    def save_results_geojson(self):
        """
        Save the OD results to a GeoJSON file, preserving geometry and ordering properties as:
        origin_id, destination_id, distance_meters, duration_seconds, polyline.
        """
        features = []
        for res in self.results:
            feature = {
                "origin_id": res["origin_id"],
                "destination_id": res["destination_id"],
                "distance_meters": res["distance_meters"],
                "duration_seconds": res["duration_seconds"],
                "polyline": res["polyline"]
            }

            origin = next(
                (pair[1] for pair in self.od_pairs if pair[0] == res["origin_id"] and pair[2] == res["destination_id"]),
                None
            )
            destination = next(
                (pair[3] for pair in self.od_pairs if pair[0] == res["origin_id"] and pair[2] == res["destination_id"]),
                None
            )

            if origin and destination:
                geometry = {
                    "type": "LineString",
                    "coordinates": [
                        [origin[1], origin[0]],  # [lon, lat]
                        [destination[1], destination[0]]
                    ]
                }
            else:
                geometry = None  # fallback if not found

            features.append({
                "type": "Feature",
                "properties": feature,
                "geometry": geometry
            })

        geojson_dict = {
            "type": "FeatureCollection",
            "features": features
        }

        geojson_path = self.output_path.replace(".json", "_with_geometry.geojson")
        with open(geojson_path, "w", encoding="utf-8") as f:
            json.dump(geojson_dict, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(features)} OD results to {geojson_path}")

    def run(self):
        """Run the full OD matrix generation pipeline."""
        self.load_nodes()
        self.generate_od_pairs()
        self.request_od_metrics()
        self.save_results_geojson()
