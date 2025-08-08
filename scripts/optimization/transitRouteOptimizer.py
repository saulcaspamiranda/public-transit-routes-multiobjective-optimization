from validRouteSampler import ValidRouteSampler
from dataLoader import DataLoader
from sklearn.cluster import KMeans
from transitRouteProblem import TransitRouteProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from validRouteMutation import ValidRouteMutation
from validRouteCrossover import ValidRouteCrossover
import numpy as np
import pandas as pd
import geojson as geojson
import json as json

class TransitRouteOptimizer:
    def __init__(self, nodes_filename, od_matrix_filename, straight_stop_node_pairs_path, orbital_stop_quads_path,
                 grid_1000m_map_path, num_of_generations, population_size, num_routes_per_individual,
                 max_nodes_per_route, max_route_time_seconds, min_sample_overlap_threshold, max_sample_overlap_threshold, optimization_max_route_overlap_threshold):
        self.nodes_path = nodes_filename
        self.od_matrix_path = od_matrix_filename
        self.straight_stop_node_pairs_path = straight_stop_node_pairs_path
        self.oribital_stop_quads_path = orbital_stop_quads_path
        self.grid_1000m_map_path = grid_1000m_map_path
        self.num_of_generations = num_of_generations
        self.population_size = population_size
        self.num_routes_per_individual = num_routes_per_individual
        self.max_nodes = max_nodes_per_route  
        self.max_route_time_seconds = max_route_time_seconds
        self.min_sample_overlap_threshold = min_sample_overlap_threshold
        self.max_sample_overlap_threshold = max_sample_overlap_threshold
        self.optimization_max_route_overlap_threshold = optimization_max_route_overlap_threshold

        self.od_df = None
        self.nodes_gdf = None
        self.population_grid = None
        self.route_stop_pairs_df = None
        self.orbital_stop_quads_df = None
        self.problem = None
        self.node_near_valid_neighbors = None
        self.generator = None
        self.sampler = None
        self.algorithm = None
        self.result = None
        
    def load_data(self):
        loader = DataLoader()
        self.nodes_gdf = loader.load_nodes(self.nodes_path)
        self.od_df = loader.load_od_matrix(self.od_matrix_path)
        self.route_stop_pairs_df = loader.load_starting_node_pairs(self.straight_stop_node_pairs_path)
        self.orbital_stop_quads_df = loader.load_orbital_stop_quads(self.oribital_stop_quads_path)
        self.population_grid = loader.load_population_grid(self.grid_1000m_map_path)

    def setup_problem(self):
        print("Setting up the Transit Route Problem...")
        self.problem = TransitRouteProblem(self.od_df,
                                           self.nodes_gdf,
                                           self.population_grid,
                                           self.num_routes_per_individual,
                                           self.max_nodes,
                                           self.max_route_time_seconds,
                                           self.optimization_max_route_overlap_threshold)

    def build_neighbors(self):
        self.node_near_valid_neighbors = {}
        for (origin, dest) in self.od_df.index:
            self.node_near_valid_neighbors.setdefault(origin, set()).add(dest)

    def setup_sampler(self):
        print("Setting up the Valid Route Sampler...")
        self.sampler = ValidRouteSampler(
            neighbors=self.node_near_valid_neighbors,
            num_routes=self.problem.num_routes,
            max_nodes_per_route=self.max_nodes,
            nodes_gdf=self.nodes_gdf,
            od_df=self.od_df,
            starting_node_pairs_df=self.route_stop_pairs_df,
            orbital_stop_quads_df=self.orbital_stop_quads_df,
            max_route_time_seconds=self.max_route_time_seconds,
            min_route_overlap_threshold=self.min_sample_overlap_threshold,
            max_route_overlap_threshold=self.max_sample_overlap_threshold
        )

    def decreasing_schedule(self, gen, max_gen):
        return max(0.05, 0.6 * (1 - gen / max_gen))  # starts at 60%, goes down to 5%

    def setup_algorithm(self):
        self.algorithm = NSGA2(
            pop_size=self.population_size,
            sampling=self.sampler,
            crossover = ValidRouteCrossover(
                num_routes=self.num_routes_per_individual,
                max_nodes=self.max_nodes,
                schedule=lambda gen: self.decreasing_schedule(gen, self.num_of_generations)
            ),
            mutation=ValidRouteMutation(self.node_near_valid_neighbors, prob=0.20),
            eliminate_duplicates=False
        )

    def run_optimization(self):
        print("Running the optimization...")
        termination = get_termination("n_gen", self.num_of_generations)
        self.result = minimize(
            self.problem,
            self.algorithm,
            termination,
            seed=1,
            save_history=True,
            verbose=True
        )

    def print_results(self):
        print("\nBest individual found:")
        print(self.result.X)
        print("Objectives:", self.result.F)
    
    def save_positive_pareto_front_csv(self, filename):
        """
        Save the Pareto front objectives to a CSV file, restoring maximized values (f1 and f3),
        and include the corresponding row index. Values are rounded to integers.
        """
        if self.result is None or self.result.F is None:
            print("No results to save.")
            return

        # Restore original positive values for f1 and f3
        F_original = self.result.F.copy()
        F_original[:, 0] = -F_original[:, 0]  # f1: connectivity (was negated)
        F_original[:, 2] = -F_original[:, 2]  # f3: concurrence (was negated)

        # Round values to nearest integer
        F_rounded = np.rint(F_original).astype(int)

        # Add index column
        indices = np.arange(len(F_rounded)).reshape(-1, 1)
        data_with_index = np.hstack((indices, F_rounded))

        # Save to CSV with integer format
        np.savetxt(
            filename,
            data_with_index,
            delimiter=",",
            header="index,f1_Network_Node_Connection,f2_Travel_Time,f3_Concurrence_Served",
            comments="",
            fmt=["%d", "%d", "%d", "%d"]
        )

        print(f"Pareto front with indices saved to {filename}")
    
    def cluster_pareto_solutions(self, csv_filename, output_path, n_clusters):
        """
        Cluster Pareto front solutions from CSV to find representative solutions,
        preserving original Pareto `index` column.
        """
        import pandas as pd
        import numpy as np
        from sklearn.cluster import KMeans

        # Load CSV
        df = pd.read_csv(csv_filename)

        # Extract objective values
        X = df[["f1_Network_Node_Connection", "f2_Travel_Time", "f3_Concurrence_Served"]].values

        # Apply KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        labels = kmeans.fit_predict(X)

        # Find closest point to each cluster centroid
        representatives = []
        for i in range(n_clusters):
            cluster_df = df[labels == i]
            cluster_points = cluster_df[["f1_Network_Node_Connection", "f2_Travel_Time", "f3_Concurrence_Served"]].values
            centroid = kmeans.cluster_centers_[i]
            distances = np.linalg.norm(cluster_points - centroid, axis=1)
            closest_row = cluster_df.iloc[np.argmin(distances)]
            representatives.append(closest_row)

        # Save result
        pd.DataFrame(representatives).to_csv(output_path, index=False)

    def export_all_solutions_to_json(self, output_filename):
        """
        Creates a json file with all solutions, their routes and their route nodes
        """
        results = []
        for sol_idx, solution in enumerate(self.result.X):
            route_size = self.problem.max_nodes
            od_routes = []
            for r in range(self.problem.num_routes):
                route_osmids = []
                od_ids = []
                prev_osmid = None
                for n in range(self.problem.max_nodes):
                    idx = r * route_size + n
                    osmid = int(solution[idx])
                    if osmid != -1:
                        route_osmids.append(osmid)
                        if prev_osmid is not None:
                            od_id = f"{prev_osmid}-{osmid}"
                            if (prev_osmid, osmid) in self.od_df.index:
                                od_ids.append(od_id)
                        prev_osmid = osmid
                if od_ids:
                    od_routes.append(od_ids)
            if len(od_routes) == self.problem.num_routes:
                results.append({
                    "solution_index": sol_idx,
                    "od_routes": od_routes
                })
        with open(output_filename, "w") as f:
            json.dump(results, f, indent=2)
        print(f"All OD ID lists exported to {output_filename}")
    
    def run(self):
        self.load_data()
        self.setup_problem()
        self.build_neighbors()
        self.setup_sampler()
        self.setup_algorithm()
        self.run_optimization()
        self.print_results()
        self.save_positive_pareto_front_csv(f"data_files/result_files/positive_pareto_front_{self.num_routes_per_individual}_routes.csv")
        self.cluster_pareto_solutions(f"data_files/result_files/positive_pareto_front_{self.num_routes_per_individual}_routes.csv",
                                      f"data_files/result_files/clustered_pareto_solutions_{self.num_routes_per_individual}_routes.csv", 10)
        self.export_all_solutions_to_json(f"data_files/result_files/pareto_solutions_{self.num_routes_per_individual}_routes.json")
