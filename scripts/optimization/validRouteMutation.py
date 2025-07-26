from pymoo.core.mutation import Mutation
import random

class ValidRouteMutation(Mutation):
    def __init__(self, neighbors, prob):
        super().__init__()
        self.neighbors = neighbors
        self.prob = prob

    def _do(self, problem, X, **kwargs):
        X = X.copy()
        num_routes = problem.num_routes
        max_nodes = problem.max_nodes

        for i in range(X.shape[0]):
            for r in range(num_routes):
                for n in range(1, max_nodes - 1):  # leave first and last node untouched for stability
                    idx = r * max_nodes + n
                    prev_idx = r * max_nodes + (n - 1)
                    next_idx = r * max_nodes + (n + 1)

                    prev_osmid = int(X[i, prev_idx])
                    next_osmid = int(X[i, next_idx])

                    if X[i, idx] == -1 or next_osmid == -1:
                        break  # stop at padding

                    if random.random() < self.prob:
                        # Only consider candidates that preserve connection to both prev and next
                        candidates = [
                            dest for (origin, dest) in problem.od_df.index
                            if origin == prev_osmid and (dest, next_osmid) in problem.od_df.index
                        ]
                        if candidates:
                            new_osmid = random.choice(candidates)
                            X[i, idx] = new_osmid

        return X