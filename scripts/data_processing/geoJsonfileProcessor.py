import geopandas as gpd
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from rasterio.mask import mask
from frechetdist import frdist
from shapely.geometry import LineString
from shapely.geometry import Point
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

class GeoJsonFileProcessor:
    """
    Class with functions to update geojson files data.
    """
    
    def readGeoJsonFile(pathToFile):
        gdf = gpd.read_file(pathToFile)
        return gdf

    def geoJsonToShp(geoJsonFilePath, shpFilePath):
        print("RESULT FILE PATH:" + shpFilePath)
        gdf = gpd.read_file(geoJsonFilePath)
        gdf.to_file(shpFilePath)
        print("GeoJson to Shp conversion completed for file: " + geoJsonFilePath)
        
    def add_sequential_id_column(self, geoJsonFilePath, outputFilePath, id_label="custom_id"):
        """
        Adds a sequential ID column (starting from 1) with the given label to the GeoJSON features.
        Saves the result to outputFilePath.
        """
        gdf = gpd.read_file(geoJsonFilePath)
        gdf.insert(0, id_label, range(1, len(gdf) + 1))  # Insert as first column
        gdf.to_file(outputFilePath, driver="GeoJSON")
        print(f"Added '{id_label}' column and saved to {outputFilePath}")
        
    def round_up_and_reorder_column(self, geoJsonFilePath, outputFilePath, col_name, position):
        """
        Rounds the specified column to integers, reorders it to the given position,
        and saves the result as a GeoJSON with geometry preserved.
        """
        gdf = gpd.read_file(geoJsonFilePath)

        if col_name not in gdf.columns:
            print(f"Column '{col_name}' not found in the GeoJSON file.")
            return

        # Round values
        gdf[col_name] = np.round(gdf[col_name]).astype(int)

        # Extract and remove the column
        col_data = gdf.pop(col_name)

        # Insert it at the desired position (before geometry)
        columns = list(gdf.columns)
        if "geometry" in columns:
            columns.remove("geometry")
        columns.insert(position, col_name)
        columns.append("geometry")

        # Rebuild GeoDataFrame with reordered columns
        gdf[col_name] = col_data
        gdf = gdf[columns]

        # Save to GeoJSON
        gdf.to_file(outputFilePath, driver="GeoJSON")
        print(f"Rounded and moved '{col_name}' to position {position}. Saved to {outputFilePath}.")

    def clip_raster_to_geojson(raster_path, geojson_path, output_path):
        """
        Clips a raster file to the boundaries defined in a GeoJSON file.
        """
        # Open raster to get its CRS
        with rasterio.open(raster_path) as src:
            raster_crs = src.crs

            # Read and reproject GeoJSON to raster CRS
            gdf = gpd.read_file(geojson_path)
            if gdf.crs != raster_crs:
                gdf = gdf.to_crs(raster_crs)

            geoms = [geom.__geo_interface__ for geom in gdf.geometry]

            # Clip
            out_image, out_transform = mask(src, geoms, crop=True)
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })

            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(out_image)

        print(f"Clipped raster saved to {output_path}")

    def filter_routes_within_border(routes_geojson_path, border_geojson_path, output_geojson_path):
        """
        Filters routes whose LineString geometry falls within the border polygon.
        Saves the filtered routes to output_geojson_path.
        """
        # Load routes and border
        routes_gdf = gpd.read_file(routes_geojson_path)
        border_gdf = gpd.read_file(border_geojson_path)

        # Ensure CRS match
        if routes_gdf.crs != border_gdf.crs:
            border_gdf = border_gdf.to_crs(routes_gdf.crs)

        # Dissolve border polygons into a single geometry (if needed)
        border_union = border_gdf.unary_union

        # Filter routes whose geometry is (fully) within the border
        filtered_routes = routes_gdf[routes_gdf.geometry.within(border_union)]

        # Save to GeoJSON
        filtered_routes.to_file(output_geojson_path, driver="GeoJSON")
        print(f"Filtered routes saved to {output_geojson_path}")

    def clusterRoutesBasedOnSpatialProximity(routes_geojson_path, output_geojson_path, n_clusters):
        """
        Clusters routes based on spatial proximity of their start and end points.
        """
        gdf = gpd.read_file(routes_geojson_path)

        # Prepare features: start and end coordinates only
        features = []
        valid_indices = []
        for idx, geom in enumerate(gdf.geometry):
            if not isinstance(geom, LineString):
                continue
            coords = np.array(geom.coords)
            start = coords[0]
            end = coords[-1]
            feature = np.concatenate([start, end])
            features.append(feature)
            valid_indices.append(idx)

        features = np.array(features)
        features = StandardScaler().fit_transform(features)

        # Agglomerative clustering with a fixed number of clusters
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        labels = clustering.fit_predict(features)

        # Assign cluster labels only to valid rows
        gdf['shape_cluster'] = -1
        for idx, label in zip(valid_indices, labels):
            gdf.at[idx, 'shape_cluster'] = label

        gdf.to_file(output_geojson_path, driver='GeoJSON')
        print(gdf[['linea', 'shape_cluster']])

        # Optional: visualize clusters
        for cluster in gdf['shape_cluster'].unique():
            subset = gdf[gdf['shape_cluster'] == cluster]
            for geom in subset.geometry:
                x, y = geom.xy
                plt.plot(x, y, label=f'Cluster {cluster}')
        plt.legend()
        plt.show()

    def select_representative_route_per_cluster(clustered_routes_geojson_path, output_geojson_path, cluster_col='shape_cluster'):
        """
        For each cluster, selects the most representative route (medoid) using Discrete Fréchet distance.
        Saves the representative routes to output_geojson_path.
        """
        gdf = gpd.read_file(clustered_routes_geojson_path)
        print(gdf.geom_type.value_counts())
        representatives = []

        for cluster_id in gdf[cluster_col].unique():
            cluster_gdf = gdf[gdf[cluster_col] == cluster_id]
            coords_list = []
            valid_indices = []
            for idx, geom in enumerate(cluster_gdf.geometry):
                # Only process valid LineStrings
                if isinstance(geom, LineString) and len(geom.coords) > 1:
                    coords_2d = []
                    for c in geom.coords:
                        # Force coordinate to be exactly 2D by slicing
                        if isinstance(c, (list, tuple)) and len(c) >= 2:
                            coords_2d.append(tuple(c[:2]))
                        else:
                            print(f"Warning: Skipping malformed coordinate {c} in feature {idx}")
                    # Only add if all coordinates are valid
                    # ...existing code...
            if len(coords_2d) == len(geom.coords):
                # Check all points are 2D
                if all(len(pt) == 2 for pt in coords_2d):
                    coords_list.append(coords_2d)
                    valid_indices.append(idx)
                else:
                    print(f"Warning: Skipping LineString with non-2D coordinates at index {idx}: {coords_2d}")
            else:
                print(f"Warning: Skipping LineString with malformed coordinates at index {idx}")
                
            n = len(coords_list)
            if n == 0:
                continue
            if n == 1:
                representatives.append(cluster_gdf.iloc[valid_indices[0]])
                continue

            # Compute pairwise Fréchet distances
            dist_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    d = frdist(coords_list[i], coords_list[j])
                    dist_matrix[i, j] = dist_matrix[j, i] = d

            avg_distances = dist_matrix.mean(axis=1)
            medoid_idx = np.argmin(avg_distances)
            representatives.append(cluster_gdf.iloc[valid_indices[medoid_idx]])

        rep_gdf = gpd.GeoDataFrame(representatives, crs=gdf.crs)
        rep_gdf.to_file(output_geojson_path, driver="GeoJSON")
        print(f"Saved representative routes to {output_geojson_path}")

    def remove_osmid_duplicates(filePath, outputFilePath):
        """
        Removes duplicate records based on the 'osmid' column, keeping the first occurrence.
        Returns a GeoDataFrame with unique 'osmid' values as index.
        """
        gdf = GeoJsonFileProcessor.readGeoJsonFile(filePath)
        gdf = gdf.drop_duplicates(subset="osmid", keep="first")
        gdf.to_file(outputFilePath, driver="GeoJSON")
        return gdf
    
    def filter_od_pairs_by_nodes(od_matrix_path, nodes_path, output_path):
        """
        Removes OD pairs from the OD matrix GeoJSON file where origin_id or destination_id
        do not match any osmid in the nodes GeoJSON file. Saves the filtered OD matrix to output_path.
        """
        import geopandas as gpd

        # Load nodes and get valid osmids as a set
        nodes_gdf = gpd.read_file(nodes_path)
        valid_osmids = set(nodes_gdf["osmid"])

        # Load OD matrix
        od_gdf = gpd.read_file(od_matrix_path)

        # Filter OD pairs
        filtered_od_gdf = od_gdf[
            od_gdf["origin_id"].isin(valid_osmids) & od_gdf["destination_id"].isin(valid_osmids)
        ]

        # Save filtered OD matrix
        filtered_od_gdf.to_file(output_path, driver="GeoJSON")
        print(f"Filtered OD matrix saved to {output_path} with {len(filtered_od_gdf)} pairs.")

        return filtered_od_gdf
    
    def verify_od_pairs_with_nodes(od_matrix_path, nodes_path):
        """
        Verifies that all origin_id and destination_id values in the OD matrix
        exist in the nodes GeoDataFrame's osmid column.
        
        Returns True if all OD pairs are valid, False otherwise.
            dict: A dictionary with lists of missing origin_ids and destination_ids.
        """
        nodes_gdf = gpd.read_file(nodes_path)
        od_gdf = gpd.read_file(od_matrix_path)
        
        valid_osmids = set(nodes_gdf["osmid"])
        missing_origins = od_gdf.loc[~od_gdf["origin_id"].isin(valid_osmids), "origin_id"].unique().tolist()
        missing_destinations = od_gdf.loc[~od_gdf["destination_id"].isin(valid_osmids), "destination_id"].unique().tolist()

        all_valid = len(missing_origins) == 0 and len(missing_destinations) == 0

        if not all_valid:
            print(f"Missing origin_ids: {missing_origins}")
            print(f"Missing destination_ids: {missing_destinations}")
        else:
            print("All OD pairs are valid with respect to the nodes GeoDataFrame.")

        return all_valid, {"missing_origins": missing_origins, "missing_destinations": missing_destinations}
    
    def print_top_10_percent_concurrence_sorted(geojson_path):
        """
        Prints the top 10% concurrenceTotal values from a GeoJSON, sorted ascending.
        """
        gdf = gpd.read_file(geojson_path)

        if 'concurrenceTotal' not in gdf.columns:
            raise ValueError("'concurrenceTotal' column not found in GeoJSON.")

        threshold = gdf['concurrenceTotal'].quantile(0.9)

        top_values = gdf[gdf['concurrenceTotal'] >= threshold]['concurrenceTotal']
        top_values_sorted = top_values.sort_values()

        print("Top 10% concurrenceTotal values (ascending):")
        print(top_values_sorted.to_string(index=False))
        
    def check_osmid_duplicates(geojson_path):
        """
        Checks for duplicate 'osmid' values in a GeoJSON file.
        """
        gdf = gpd.read_file(geojson_path)

        if 'osmid' not in gdf.columns:
            raise ValueError("'osmid' column not found in the GeoJSON.")

        duplicated = gdf[gdf.duplicated(subset='osmid', keep=False)]

        if duplicated.empty:
            print("No duplicate 'osmid' values found.")
        else:
            print(f"Found {len(duplicated)} duplicated entries based on 'osmid':")
            print(duplicated[['osmid']].drop_duplicates().sort_values(by='osmid'))

        return duplicated
    
    def clean_and_reorder_geojson(input_path, output_path):
        """
        Load a GeoJSON, reorder columns (keeping geometry), and save to a new GeoJSON.
        """
        # Load the GeoJSON file
        gdf = gpd.read_file(input_path)

        # Desired property order
        desired_order = [
            "origin_id",
            "destination_id",
            "distance_meters",
            "duration_seconds",
            "polyline"
        ]

        # Ensure only existing columns are included
        reordered_cols = [col for col in desired_order if col in gdf.columns]

        # Append geometry column at the end
        final_columns = reordered_cols + ["geometry"]

        # Reorder GeoDataFrame
        gdf = gdf[final_columns]

        # Save to GeoJSON
        gdf.to_file(output_path, driver="GeoJSON")

        print(f"Saved reordered GeoJSON with geometry to: {output_path}")
    
    @staticmethod
    def replace_coordinates_for_osmid(input_path, output_path, osmid_column, coordinates_column):
        # Load the GeoJSON file
        gdf = gpd.read_file(input_path)

        # Parse coordinates
        try:
            lat_str, lon_str = coordinates_column.strip().split(",")
            lat = float(lat_str)
            lon = float(lon_str)
        except Exception as e:
            raise ValueError(f"Invalid coordinate format: {coordinates_column}") from e

        # Ensure osmid_column is a string and convert column to string to compare
        osmid_column = str(osmid_column)
        gdf["osmid_str"] = gdf["osmid"].astype(str)

        # Find the matching row
        mask = gdf["osmid_str"] == osmid_column
        if not mask.any():
            raise ValueError(f"osmid '{osmid_column}' not found in file: {input_path}")

        # Update geometry and coordinate columns
        new_point = Point(lon, lat)
        gdf.loc[mask, "geometry"] = new_point
        gdf.loc[mask, "x"] = lon
        gdf.loc[mask, "y"] = lat

        # Drop helper column and save result
        gdf.drop(columns=["osmid_str"], inplace=True)
        gdf.to_file(output_path, driver="GeoJSON")
        print(f"Updated node {osmid_column} saved to: {output_path}")        
        
processor = GeoJsonFileProcessor()

GeoJsonFileProcessor.clip_raster_to_geojson(
    "data_files/base_files/GHS_POP_E2030_GLOBE_R2023A_54009_100_V1_0_R12_C12.tif",
    "data_files/base_files/cochabamba_boundaries.geojson",
    "data_files/base_files/GHS_POP_E2030_clipped_to_Cercado.tif"
)

GeoJsonFileProcessor.filter_routes_within_border(
    "data_files/base_files/cochabamba_official_routes/official_routes_cochabamba_2018.geojson",
    "data_files/base_files/cochabamba_boundaries.geojson",
    "data_files/result_files/cochabamba_official_routes/filtered_official_routes_to_cochabamba.geojson"
)

GeoJsonFileProcessor.clusterRoutesBasedOnSpatialProximity(
    "data_files/result_files/cochabamba_official_routes/filtered_official_routes_to_cochabamba.geojson",
    "data_files/result_files/cochabamba_official_routes/clustered_routes_to_cochabamba.geojson",
    n_clusters=20
)

GeoJsonFileProcessor.select_representative_route_per_cluster(
    "data_files/result_files/cochabamba_official_routes/clustered_routes_to_cochabamba.geojson",
    "data_files/result_files/cochabamba_official_routes/n13_clusteR_representative_routes_cochabamba.geojson"
)