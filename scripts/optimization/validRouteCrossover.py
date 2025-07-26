from pymoo.core.crossover import Crossover
import numpy as np
import random

class ValidRouteCrossover(Crossover):
    """
    Custom crossover that takes a percentage of routes from two parents and swaps them in their children.
    """
    def __init__(self, num_routes, max_nodes, schedule=None):
        super().__init__(2, 2)
        self.num_routes = num_routes
        self.max_nodes = max_nodes
        self.schedule = schedule if schedule else (lambda gen: 0.2)  # default to 20%

    def _do(self, problem, X, **kwargs):
        X = np.transpose(X, (1, 0, 2))  # shape: (n_matings, 2, n_var)
        n_matings, _, n_var = X.shape
        children = np.empty((2, n_matings, n_var), dtype=X.dtype)

        route_size = self.max_nodes

        generation = kwargs.get("algorithm").n_gen  # <-- current generation

        swap_fraction = self.schedule(generation)
        num_to_swap = max(1, int(self.num_routes * swap_fraction))

        for k in range(n_matings):
            p1, p2 = X[k, 0], X[k, 1]
            child1 = p1.copy()
            child2 = p2.copy()

            routes_to_swap = random.sample(range(self.num_routes), k=num_to_swap)

            for r in routes_to_swap:
                route_offset = r * route_size
                child1[route_offset:route_offset + route_size] = p2[route_offset:route_offset + route_size]
                child2[route_offset:route_offset + route_size] = p1[route_offset:route_offset + route_size]

            children[0, k, :] = child1
            children[1, k, :] = child2

        return children