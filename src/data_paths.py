from pathlib import Path
import os

DATA_BASE = Path(os.getcwd())/'data' # Or specify accurate path for the project root

# Replace with your own paths to Topographical database

# Path to Finnish transport infrastructure data
PATH_TO_BEACONS = DATA_BASE/'vaylavirasto/turvalaitteetPoint.shp'

PATH_TO_SEA = DATA_BASE/'mtk/seas.parquet'
PATH_TO_LAKES = DATA_BASE/'mtk/lakes.parquet'
PATH_TO_RIVERS = DATA_BASE/'mtk/rivers.parquet'
PATH_TO_ROCK_AREAS = DATA_BASE/'mtk/water_rock_areas.parquet'
PATH_TO_ROCK_POINTS = DATA_BASE/'mtk/water_rock_points.parquet'
PATH_TO_WINDMILLS = DATA_BASE/'mtk/windmills.parquet'


# Optional presets
PATH_TO_MTK_DATA = DATA_BASE/'mtk'
PATH_TO_ARCHI_WATERS = DATA_BASE/'mtk/archipelago_waters.gpkg'
PATH_TO_GOF_WATERS = DATA_BASE/'mtk/gof_waters.gpkg'

# Hand drawn layer for fisheries
PATH_TO_FISHERIES = DATA_BASE/'fisheries_3067.gpkg'
