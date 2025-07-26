import numpy as np
import pandas as pd
import geopandas as gpd
from pymoo.core.problem import Problem
from itertools import combinations
from math import ceil

class TransitRouteProblem(Problem):
    
    attribute_concurrenceTotal = "concurrenceTotal"
    attribute_duration_seconds = "duration_seconds"
    attribute_distance_meters = "distance_meters"
    attribute_geometry = "geometry"
    
    def __init__(
        self,
        od_df: pd.DataFrame,
        nodes_gdf: gpd.GeoDataFrame,
        population_grid: gpd.GeoDataFrame,
        num_routes: int,
        max_nodes: int,
        max_route_time_seconds: int,
        route_jaccard_overlap_threshold_for_penalty: float,
    ):
        self.od_df = od_df
        self.nodes_gdf = nodes_gdf
        self.population_grid = population_grid
        self.num_routes = num_routes
        self.max_nodes = max_nodes
        self.max_route_time_seconds = max_route_time_seconds
        self.route_jaccard_overlap_threshold_for_penalty = route_jaccard_overlap_threshold_for_penalty
        self.node_conc_dict, self.od_dict, self.od_geom_dict = self.loadDictionaries()
        self.constrains = num_routes * 3 + 2
        n_var = num_routes * max_nodes
        xl = np.tile(np.array([-1, 0.0, 0.0]), num_routes * max_nodes)
        xu = np.tile(np.array([np.iinfo(np.int64).max, 1e10, 1e10]), num_routes * max_nodes)
        
        super().__init__(
            n_var=n_var,
            n_obj=3,
            n_constr=self.constrains,
            xl=xl,
            xu=xu,
            elementwise_evaluation=False,
            type_var=int
        )

    def loadDictionaries(self):
        """
        Create data dictionaries for faster searching.
        """
        # Load node-level concurrence values
        node_conc_dict = dict(zip(self.nodes_gdf.index, self.nodes_gdf[self.attribute_concurrenceTotal]))

        # Load OD dictionary (duration & distance)
        od_dict = {
            (int(origin), int(dest)): {
                self.attribute_duration_seconds: row[self.attribute_duration_seconds],
                self.attribute_distance_meters: row[self.attribute_distance_meters]
            }
            for (origin, dest), row in self.od_df[[self.attribute_duration_seconds, self.attribute_distance_meters]].iterrows()
        }

        # Load geometry dictionary, filtering out missing geometries
        od_geom_dict = {
            (int(origin), int(dest)): geom
            for (origin, dest), geom in self.od_df[self.attribute_geometry].items()
            if pd.notnull(geom)
        }

        return node_conc_dict, od_dict, od_geom_dict

    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluates individuals in a generation according to the objective and constrains.
        """
        n_pop = X.shape[0]
        max_percentage_orbital_routes = 0.3
        jaccard_threshold = self.route_jaccard_overlap_threshold_for_penalty

        F = np.zeros((n_pop, 3))  # [f1, f2, f3]
        G = np.zeros((n_pop, self.constrains))

        X_reshaped = X.reshape(n_pop, self.num_routes, self.max_nodes).astype(int)
        valid_mask = X_reshaped != -1

        route_times = np.zeros((n_pop, self.num_routes))
        route_concurrences = np.zeros((n_pop, self.num_routes))
        route_lengths = np.sum(valid_mask, axis=2)
        unique_constraints = np.zeros((n_pop, self.num_routes))

        for pop_idx in range(n_pop):
            node_connection_pairs = set()
            orbital_count = 0
            individual_routes = []

            for r in range(self.num_routes):
                route_osmids = X_reshaped[pop_idx, r][valid_mask[pop_idx, r]]

                if len(route_osmids) > 1:
                    individual_routes.append(set(route_osmids))

                    od_keys = [(route_osmids[i], route_osmids[i + 1]) for i in range(len(route_osmids) - 1)]
                    route_times[pop_idx, r] = sum(
                        self.od_dict.get(od, {self.attribute_duration_seconds: 0})[self.attribute_duration_seconds] for od in od_keys
                    )

                    route_concurrences[pop_idx, r] = sum(
                        self.node_conc_dict.get(n, 0) for n in route_osmids
                    )

                    route_osmids_unique = (
                        route_osmids[:-1] if route_osmids[0] == route_osmids[-1] else route_osmids
                    )
                    unique_constraints[pop_idx, r] = 0 if len(set(route_osmids_unique)) == len(route_osmids_unique) else 1

                    if route_osmids[0] == route_osmids[-1]:
                        orbital_count += 1

                    for a, b in combinations(route_osmids, 2):
                        node_connection_pairs.add(frozenset((a, b)))

            # Jaccard penalty as constraint
            jaccard_penalty = self._penalize_jaccard_overlap(individual_routes, jaccard_threshold)
            
            # Objectives
            F[pop_idx, 0] = -len(node_connection_pairs)
            F[pop_idx, 1] = route_times[pop_idx].sum()
            F[pop_idx, 2] = -route_concurrences[pop_idx].sum()

            # Constraints, 3 per route
            G[pop_idx, :self.num_routes * 3:3] = route_times[pop_idx] - self.max_route_time_seconds
            G[pop_idx, 1:self.num_routes * 3:3] = route_lengths[pop_idx] - self.max_nodes
            G[pop_idx, 2:self.num_routes * 3:3] = unique_constraints[pop_idx]

            # Global constraints (at the end of G)
            max_orbital_allowed = ceil(self.num_routes * max_percentage_orbital_routes)
            G[pop_idx, -2] = max(0, orbital_count - max_orbital_allowed)
            G[pop_idx, -1] = jaccard_penalty
            
        out["F"] = F
        out["G"] = G

    def _penalize_jaccard_overlap(self, routes, threshold):
        penalty = 0
        for route_a, route_b in combinations(routes, 2):
            union = route_a | route_b
            if not union:
                continue
            jaccard = len(route_a & route_b) / len(union)
            if jaccard > threshold:
                penalty += (jaccard - threshold)
        return penalty