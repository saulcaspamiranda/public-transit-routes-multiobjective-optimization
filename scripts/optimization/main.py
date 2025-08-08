from transitRouteOptimizer import TransitRouteOptimizer
from routeGeoJsonProcessor import RouteGeoJSONProcessor

class Main:

    def runOptimizer():        
        optimizer = TransitRouteOptimizer(
            nodes_filename="data_files/base_files/nodes6_with_200m_concurrence_assigned_centered.geojson",
            od_matrix_filename="data_files/base_files/origin_destinations_matrix_6.geojson",
            straight_stop_node_pairs_path="data_files/base_files/straight_stop_node_pairs.geojson",
            orbital_stop_quads_path="data_files/base_files/orbital_stop_quads.geojson",
            grid_1000m_map_path="data_files/base_files/cochabamba_1000m_population_grid_epsg4326.geojson",
            num_of_generations=200,
            population_size=300,
            num_routes_per_individual=13,
            max_nodes_per_route=60,
            max_route_time_seconds= 70 * 60,  # minutes * seconds in a minute
            min_sample_overlap_threshold = 0.3,
            max_sample_overlap_threshold = 0.8,
            optimization_max_route_overlap_threshold = 0.5 # greater than the set % of overlap in routes for an individual is penalized
        )
        optimizer.run()

    def decodeRouteLines():
        RouteGeoJSONProcessor.decode_polylines_for_routes_to_geojson(
            "data_files/result_files/pareto_solutions_13_routes.json",
            "data_files/base_files/origin_destinations_matrix_6.geojson",
            "data_files/solutions/n13RoutesSolution",
            "data_files/base_files/nodes6_with_200m_concurrence_assigned_centered.geojson"
        )

Main.runOptimizer()
Main.decodeRouteLines()