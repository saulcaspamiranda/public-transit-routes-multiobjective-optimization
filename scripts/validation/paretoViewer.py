import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np

class ParetoViewer:
    """
    Class to display the pareto solution objective values in pairs or all together.
    """
    
    @staticmethod
    def plot_pareto_front_3d_from_csv(filename, obj_columns=("f1_Network_Node_Connection", "f2_Travel_Time", "f3_Concurrence_Served")):
        
        """
        Plot a 3D Pareto front from a CSV file with columns 'f1', 'f2', 'f3'.
        """
        filename = filename
        df = pd.read_csv(filename)

        x_col, y_col, z_col = obj_columns
        x = df[x_col]
        y = df[y_col]
        z = df[z_col]

        fig = go.Figure(data=[go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(size=5, color=x, colorscale='Viridis'),
            text=[
                f"Index: {idx}"
                for idx, xi, yi, zi in zip(df['index'], x, y, z)
            ]
        )])

        fig.update_layout(
            scene=dict(
                xaxis_title=" f1 Network Node Connection",
                yaxis_title=" f2 Travel Time [seconds]",
                zaxis_title=" f3 Concurrence Served"
            ),
            title='Pareto Front (f1, f2, f3)'
        )

        fig.show()

    @staticmethod
    def plot_pareto_front_from_csv(filename, obj_indices, labels=None):
        """
        Plot a 2D Pareto front from a CSV file for any pair of objectives,
        """
        # Skip the index column (first column)
        data = np.genfromtxt(filename, delimiter=",", skip_header=1)[:, 1:]  # <-- Skip index column

        x_idx, y_idx = obj_indices

        default_labels = [
            "f1 Network Node Connection",
            "f2 Travel Time",
            "f3 Concurrence Served"
        ]
        
        xlabel = labels[0] if labels else default_labels[x_idx]
        ylabel = labels[1] if labels else default_labels[y_idx]

        x = data[:, x_idx]
        y = data[:, y_idx]

        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, c="darkblue", s=40, alpha=0.7, edgecolors='k')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f"Pareto Front: {xlabel} vs {ylabel}")
        plt.grid(True)

        # Adaptive log scaling if large range and all values positive
        if (x > 0).all() and x.max() / x.min() > 100:
            plt.xscale("log")
        if (y > 0).all() and y.max() / y.min() > 100:
            plt.yscale("log")

        plt.tight_layout()
        plt.show()


ParetoViewer.plot_pareto_front_from_csv("data_files/result_files/positive_pareto_front_13_routes.csv", obj_indices=(0, 1)) # Plot f1 vs f2
ParetoViewer.plot_pareto_front_from_csv("data_files/result_files/positive_pareto_front_13_routes.csv", obj_indices=(0, 2)) # f1 vs f3
ParetoViewer.plot_pareto_front_from_csv("data_files/result_files/positive_pareto_front_13_routes.csv", obj_indices=(1, 2)) # f2 vs f3
ParetoViewer.plot_pareto_front_3d_from_csv("data_files/result_files/positive_pareto_front_13_routes.csv")
ParetoViewer.plot_pareto_front_3d_from_csv("data_files/result_files/clustered_pareto_solutions_13_routes.csv")
