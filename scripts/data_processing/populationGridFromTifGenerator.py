import os
import numpy as np
import geopandas as gpd
import rasterio
import rasterio.mask
from shapely.geometry import box, mapping
from shapely.geometry import MultiPolygon
from rasterstats import zonal_stats
from rasterio.warp import calculate_default_transform, reproject, Resampling

class PopulationGridFromTifGenerator:
    """
    Generates a population grid from a raster and boundary shapefile.
    """

    def __init__(
        self,
        original_raster: str,
        boundary_path: str,
        output_geojson: str,
        clipped_raster_path: str,
        projected_crs: str,
        output_crs: str,
        grid_size: int = 100,
    ):
        """
        Initialize the PopulationGridGenerator.
        """
        self.original_raster = original_raster
        self.boundary_path = boundary_path
        self.output_geojson = output_geojson
        self.clipped_raster_path = clipped_raster_path
        self.projected_crs = projected_crs
        self.output_crs = output_crs
        self.grid_size = grid_size
        self.boundary = None

    def load_and_project_boundary(self):
        """
        Load the boundary shapefile and reproject it to the analysis CRS.
        """
        self.boundary = gpd.read_file(self.boundary_path)
        self.boundary = self.boundary.to_crs(self.projected_crs)

    def clip_and_reproject_raster(self):
        """
        Clip the raster to the boundary in its original CRS, then reproject the clipped raster to the analysis CRS.
        Saves the result to self.clipped_raster_path.
        """
        if os.path.exists(self.clipped_raster_path):
            return

        # Clip in original CRS
        with rasterio.open(self.original_raster) as src:
            # Ensure boundary is in the same CRS as the raster for clipping
            boundary_orig_crs = self.boundary.to_crs(src.crs)
            geoms = [mapping(geom) for geom in boundary_orig_crs.geometry]
            out_image, out_transform = rasterio.mask.mask(src, geoms, crop=True)
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "crs": src.crs
            })
            clipped_temp_path = self.clipped_raster_path.replace(".tif", "_tempclip.tif")
            with rasterio.open(clipped_temp_path, "w", **out_meta) as dest:
                dest.write(out_image)
        # Reproject the clipped raster to the target CRS
        with rasterio.open(clipped_temp_path) as src:
            transform, width, height = calculate_default_transform(
                src.crs, self.projected_crs, src.width, src.height, *src.bounds
            )
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': self.projected_crs,
                'transform': transform,
                'width': width,
                'height': height
            })
            with rasterio.open(self.clipped_raster_path, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=self.projected_crs,
                        resampling=Resampling.nearest
                    )
        # Remove the temporary clipped file
        os.remove(clipped_temp_path)

    def create_grid(self):
        """
        Create a regular grid of polygons covering the extent of the clipped raster.
        """
        with rasterio.open(self.clipped_raster_path) as src:
            band = src.read(1)
            band = np.where(band < 0, 0, band)
            transform = src.transform
            bounds = src.bounds
            raster_crs = src.crs

            x_coords = np.arange(bounds.left, bounds.right, self.grid_size)
            y_coords = np.arange(bounds.bottom, bounds.top, self.grid_size)

            grid_cells = [
                box(x, y, x + self.grid_size, y + self.grid_size)
                for x in x_coords for y in y_coords
            ]

            grid_gdf = gpd.GeoDataFrame(geometry=grid_cells, crs=raster_crs)
        return grid_gdf, band, transform, src.nodata

    def clip_grid_to_boundary(self, grid_gdf):
        """
        Clip the generated grid to the boundary geometry.
        Returns the clipped grid.
        """
        return gpd.overlay(grid_gdf, self.boundary, how="intersection")

    def compute_population(self, grid_gdf, band, transform, nodata):
        """
        Compute the sum of raster values (population) for each grid cell.
        Returns grid with a 'population' column.
        """
        stats = zonal_stats(
            grid_gdf,
            band,
            affine=transform,
            stats=["sum"],
            nodata=nodata
        )
        grid_gdf["population"] = [
            int(round(s["sum"])) if s["sum"] is not None else 0
            for s in stats
        ]
    
        return grid_gdf[grid_gdf["population"] > 0]
    
    def simplify_geometry(self, geom):
        if isinstance(geom, MultiPolygon):
            return max(geom.geoms, key=lambda p: p.area)
        return geom

    def reproject_and_export(self, grid_gdf):
        """
        Reproject the grid to the output CRS and export it as a GeoJSON file.
        """
        grid_gdf = grid_gdf.to_crs(self.output_crs)

        # Add unique FID or ID column
        grid_gdf["FID"] = range(len(grid_gdf))

        grid_gdf.to_file(self.output_geojson, driver="GeoJSON")
        print(f"Saved population grid in {self.output_crs} to {self.output_geojson}")
        
    def run(self):
        """
        Run the full population grid generation pipeline.
        """
        self.load_and_project_boundary()
        self.clip_and_reproject_raster()
        grid_gdf, band, transform, nodata = self.create_grid()
        grid_gdf = self.clip_grid_to_boundary(grid_gdf)
        grid_gdf = self.compute_population(grid_gdf, band, transform, nodata)
        grid_gdf["geometry"] = grid_gdf["geometry"].apply(self.simplify_geometry)
        self.reproject_and_export(grid_gdf)