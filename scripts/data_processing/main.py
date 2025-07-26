from geoJsonfileProcessor import GeoJsonFileProcessor
from populationGridFromTifGenerator import PopulationGridFromTifGenerator
from nodeGenerator import NodeGenerator
from nodeUpdater import NodeUpdater
from originDestinationMatrixGenerator import OriginDestinationMatrixGenerator

class Main:
    
    def getCochabambaDataFromCercadoToShp():
        cochabambaBoundariesGeoJsonFilePath = "data_files/cochabamba_boundaries.geojson"
        shpDestinationFilePath = "data_files/result_files/shp_files/cochabamba_boundaries.shp"
        GeoJsonFileProcessor.geoJsonToShp(cochabambaBoundariesGeoJsonFilePath, shpDestinationFilePath)
        
    def generatePopulationGridFromTif(grid_size):
        original_raster = "data_files/base_files/GHS_POP_E2030_GLOBE_R2023A_54009_100_V1_0_R12_C12.tif"
        boundary_path = "data_files/result_files/shp_files/cochabamba_boundaries.shp"
        output_geojson = "data_files/base_files/cochabamba_" + str(grid_size) + "m_population_grid_epsg4326.geojson"
        projected_crs = "EPSG:32719"  # UTM zone 19S
        output_crs = "EPSG:4326"      # WGS 84 for visualization
        generator = PopulationGridFromTifGenerator(
            original_raster,
            boundary_path,
            output_geojson,
            projected_crs,
            output_crs,
            grid_size=grid_size
        )
        generator.run()
        # Generates the given meters population grid for the Cercado area
        
    def generateNodes400mApart():
        place_name = "Cercado, Cochabamba, Bolivia"
        grid_path = "data_files/base_files/cochabamba_400m_population_grid_epsg4326.geojson"
        out_dir = "data_files/filtered_nodes_400m_apart_wgs84.geojson"
        generator = NodeGenerator(
            place_name=place_name,
            grid_path=grid_path,
            min_distance=400,
            utm_crs="EPSG:32719"
        )
        generator.run(out_dir)
        # Generates nodes that are at least 400 meters apart

    def generateNodes400mApartWith200mApartInRestrictionArea():
        place_name = "Cercado, Cochabamba, Bolivia"
        grid_path = "data_files/base_files/cochabamba_400m_population_grid_epsg4326.geojson"
        restriction_area_path = "data_files/base_files/vehicular_restriction_area.geojson" 
        out_dir = "data_files/base_files/filtered_nodes_400m_apart_150m_restriction_area_wgs84.geojson"
        generator = NodeGenerator(
            place_name=place_name,
            grid_path=grid_path,
            restriction_area_path=restriction_area_path,
            min_distance=400,
            utm_crs="EPSG:32719"
        )
        generator.run_with_restriction_area(out_dir)
        # Generates nodes that are at least 400 meters apart

    def updateNodesWith200mRadiusAssignedPopulation():
        nodes_filename = "data_files/base_files/filtered_nodes_400m_apart_150m_restriction_area_wgs84.geojson"
        grid_filename = "data_files/base_files/cochabamba_100m_population_grid_epsg4326.geojson"
        output_filename = "data_files/nodes3_with_200m_150m_radius_assigned_population.geojson"

        updater = NodeUpdater(
            nodes_filename=nodes_filename,
            grid_filename=grid_filename,
            output_filename=output_filename,
            buffer_radius=200,
            utm_crs="EPSG:32719"
        )
        updater.run()
        
    def generateOriginDestinationMatrix():
        input_filename = "data_files/base_files/nodes6_with_200m_concurrence_assigned_centered.geojson"
        output_filename = "data_files/base_files/origin_destinations_matrix_6.geojson"
        api_key = "Api Key"  # Google Directions API key
        print("Generating Origin-Destination Matrix...")
        generator = OriginDestinationMatrixGenerator(
            input_filename=input_filename,
            output_filename=output_filename,
            api_key=api_key,
            max_qps=10,
            max_retries=3,
            distance_threshold_km=1.3
        )
        generator.run()
        
Main.generatePopulationGridFromTif(400)
Main.generateNodes400mApart()
Main.generatePopulationGridFromTif(1000)
Main.generateNodes400mApartWith200mApartInRestrictionArea()
Main.updateNodesWith200mRadiusAssignedPopulation()
Main.generateOriginDestinationMatrix()