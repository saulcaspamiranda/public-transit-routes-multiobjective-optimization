from pymoo.core.sampling import Sampling
import pandas as pd
import numpy as np
import random
import json
import os
import networkx as nx

class ValidRouteSampler(Sampling):
    """
    Custom sampler that generates a population with each individual having a chance to allow a random overlap threshold,
    a random weight type between travel duration and distance for route generation, and a random chance to have their route's direction inverted.
    Straight routes are generated with a middle node that is among the nodes with higher concurrence.
    Orbital routes are generated from the quad nodes provided.
    """
    def __init__(self, neighbors, num_routes, max_nodes_per_route,
                 nodes_gdf, od_df, starting_node_pairs_df, orbital_stop_quads_df, max_route_time_seconds, min_route_overlap_threshold, max_route_overlap_threshold):
        super().__init__()
        self.neighbors = neighbors
        self.num_routes = num_routes
        self.max_nodes = max_nodes_per_route
        self.nodes_gdf = nodes_gdf
        self.od_df = od_df
        self.starting_node_pairs_df = starting_node_pairs_df
        self.orbital_stop_quads_df = orbital_stop_quads_df
        self.max_route_time_seconds = max_route_time_seconds
        self.min_route_overlap_threshold = min_route_overlap_threshold
        self.max_route_overlap_threshold = max_route_overlap_threshold

        self.distance_attribute_graph_weight = "distance_meters"
        self.time_attribute_graph_weight = "duration_seconds"
        self.top_percentage_concurrence_nodes_for_middle_node = 5
        self.time_weight_chance = 0.5
        self.straight_type_of_route_chance = 0.8

        self.od_graph = ValidRouteSampler.build_od_graph(self.od_df, self.time_attribute_graph_weight)
        self.node_geom, self.top_percent_high_concurrence_middle_nodes = self.load_node_data()
        self.od_lookup = self.load_od_lookup()

    def load_node_data(self):
        top_percent = self.top_percentage_concurrence_nodes_for_middle_node
        n_top = int(len(self.nodes_gdf) * top_percent / 100)

        high_concurrence_nodes = list(
            self.nodes_gdf.sort_values("concurrenceTotal", ascending=False).head(n_top).index
        )

        node_geom = {int(idx): row.geometry for idx, row in self.nodes_gdf.iterrows()}
        return node_geom, high_concurrence_nodes

    def load_od_lookup(self):
        return lambda a, b: {
            "duration_seconds": self.od_df.loc[(a, b), "duration_seconds"]
            if (a, b) in self.od_df.index else float('inf'),
            "distance_meters": self.od_df.loc[(a, b), "distance_meters"]
            if (a, b) in self.od_df.index else float('inf')
        }

    def _do(self, problem, n_samples, **kwargs):
        """
        Generates the population with orbital and straight routes.
        """
        genome_length = self.num_routes * self.max_nodes
        population = np.full((n_samples, genome_length), fill_value=-1, dtype=int)
        print_batch = 0

        sample_idx = 0
        while sample_idx < n_samples:
            pos = 0
            individual_routes = []

            # Random overlap threshold for this individual
            overlap_threshold = random.uniform(self.min_route_overlap_threshold, self.max_route_overlap_threshold)
            route_attempts = 0

            # Try to generate valid routes for this individual
            while len(individual_routes) < self.num_routes and route_attempts < 1000:
                route_attempts += 1
                route = self.generate_type_of_route()
                if not route:
                    continue

                route_set = set(route)

                # Check overlap with already accepted routes
                if any(len(route_set & prev_route) / max(len(route_set), 1) > overlap_threshold for prev_route in individual_routes):
                    continue

                # Accept route
                individual_routes.append(route_set)

                # Pad and store into genome
                padded_route = route + [-1] * (self.max_nodes - len(route))
                population[sample_idx, pos:pos + self.max_nodes] = padded_route
                pos += self.max_nodes

                print_batch += 1
                if print_batch % 100 == 0:
                    print(f"{print_batch} total sample routes generated")

            # If not enough routes, retry individual
            if len(individual_routes) < self.num_routes:
                print(f"Retrying individual {sample_idx}: only {len(individual_routes)} valid routes.")
                continue

            # Successfully generated an individual
            sample_idx += 1

        self.save_initial_population(population.tolist())
        return population
    
    def generate_type_of_route(self):
        if random.random() < self.straight_type_of_route_chance:
            return self.generate_straight_route_from_pair_with_middle_node()
        else:
            return self.generate_orbital_route_from_quad()
    
    def generate_straight_route_from_pair_with_middle_node(self):
        weight_type = self.time_attribute_graph_weight if random.random() < self.time_weight_chance else self.distance_attribute_graph_weight
        attempts = 0
        while attempts < 100:
            # Sample a valid OD pair and middle node
            pair_row = self.starting_node_pairs_df.sample(1).iloc[0]
            origin = int(pair_row['origin_id'])
            destination = int(pair_row['destination_id'])
            
            # 50% chance to swap origin and destination
            if random.random() < 0.5:
                origin, destination = destination, origin
            
            middle = random.choice(self.top_percent_high_concurrence_middle_nodes)

            try:
                origin_geom = self.node_geom[origin]
                middle_geom = self.node_geom[middle]
                dest_geom = self.node_geom[destination]
            except KeyError:
                attempts += 1
                continue

            # Approximate length to quickly reject bad middle nodes
            d1 = origin_geom.distance(middle_geom)
            d2 = middle_geom.distance(dest_geom)
            d_total = d1 + d2
            if d_total == 0:
                attempts += 1
                continue

            # Build subpaths
            path1 = self.generate_path(weight_type, origin, middle)
            if not path1 or len(path1) < 2:
                attempts += 1
                continue

            path2 = self.generate_path(weight_type, middle, destination)
            if not path2 or len(path2) < 2:
                attempts += 1
                continue

            # Combine and validate with time check
            combined = path1[:-1] + path2
            route_time = 0
            feasible = True

            for a, b in zip(combined[:-1], combined[1:]):
                od_data = self.od_lookup(a, b)
                if not od_data:
                    feasible = False
                    break
                route_time += od_data["duration_seconds"]
                if route_time > self.max_route_time_seconds:
                    feasible = False
                    break

            if feasible:
                return combined

            attempts += 1

        # Fallback default route (use a better strategy if you want)
        fallback = [self.top_percent_high_concurrence_middle_nodes[0]] * min(3, self.max_nodes)
        return fallback

    def generate_orbital_route_from_quad(self):
        weight_type = self.time_attribute_graph_weight if random.random() < self.time_weight_chance else self.distance_attribute_graph_weight
        attempts = 0

        while attempts < 100:
            quad_row = self.orbital_stop_quads_df.sample(1).iloc[0]
            quad_osmids = quad_row["osmid"]
            
            # 50% chance to reverse the order of the quad
            if random.random() < 0.5:
                quad_osmids.reverse()

            if not isinstance(quad_osmids, list) or len(quad_osmids) != 4 or len(set(quad_osmids)) < 4:
                attempts += 1
                continue

            combined = []
            total_time = 0
            feasible = True

            for i in range(4):
                a = quad_osmids[i]
                b = quad_osmids[(i + 1) % 4]

                sub_path = self.generate_path(weight_type, a, b)
                if not sub_path or len(sub_path) < 2:
                    feasible = False
                    break

                # Avoid duplicating nodes at joins
                combined.extend(sub_path if i == 0 else sub_path[1:])

                # Check OD times and accumulate route time
                for u, v in zip(sub_path[:-1], sub_path[1:]):
                    od_data = self.od_lookup(u, v)
                    if not od_data:
                        feasible = False
                        break
                    total_time += od_data["duration_seconds"]
                    if total_time > self.max_route_time_seconds:
                        feasible = False
                        break

                if not feasible:
                    break

            if feasible:
                return combined

            attempts += 1

        # Fallback route
        return [self.top_percent_high_concurrence_middle_nodes[0]] * min(3, self.max_nodes)

    def generate_path(self, weight_type, start, goal):
        try:
            path = nx.shortest_path(self.od_graph, source=start, target=goal, weight=weight_type)
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def build_od_graph(od_df, node_graph_time_weight_attribute):
        G = nx.DiGraph()
        node_graph_weight_attribute = node_graph_time_weight_attribute
    
        if isinstance(od_df.index, pd.MultiIndex):
            df = od_df.reset_index()
        else:
            df = od_df

        for _, row in df.iterrows():
            origin = row["origin_id"]
            dest = row["destination_id"]
            weight = row[node_graph_weight_attribute]
            G.add_edge(origin, dest, weight=weight)
            
        return G

    def save_initial_population(self, population_array):
        """
        Saves the population route nodes list in a json file.
        """
        output_path=f"data_files/result_files/initial_population_{self.num_routes}_routes.json"
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))

        solutions = []
        for sol_idx, individual in enumerate(population_array):
            od_routes = []
            for r in range(self.num_routes):
                route = []
                start = r * self.max_nodes
                end = start + self.max_nodes

                route_nodes = individual[start:end]

                for i in range(len(route_nodes) - 1):
                    origin_id = int(route_nodes[i])
                    dest_id = int(route_nodes[i + 1])
                    if origin_id == -1 or dest_id == -1:
                        break
                    route.append(f"{origin_id}-{dest_id}")

                if route:
                    od_routes.append(route)

            solutions.append({
                "solution_index": sol_idx,
                "od_routes": od_routes
            })

        with open(output_path, "w") as f:
            json.dump(solutions, f, indent=2)

        print(f"Initial population saved to: {output_path}")