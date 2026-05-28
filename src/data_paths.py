from pathlib import Path
import os

DATA_BASE = Path(os.getcwd())/'data' # Or specify accurate path for the project root

# Replace with your own paths to Topographical database
PATH_TO_WATERS = DATA_BASE/'mtk/MTK-vakavesi_22-03-03.gpkg'
PATH_TO_RIVERS = DATA_BASE/'mtk/MTK-virtavesi_22-03-03.gpkg'
PATH_TO_OTHER = DATA_BASE/'mtk/MTK-muut_22-03-03.gpkg'
PATH_TO_BUILDINGS = DATA_BASE/'mtk/MTK-rakennus_22-03-03.gpkg'

# Path to Finnish transport infrastructure data
PATH_TO_BEACONS = DATA_BASE/'vaylavirasto/turvalaitteetPoint.shp'

# Optional presets
PATH_TO_MTK_DATA = DATA_BASE/'mtk'
PATH_TO_ARCHI_WATERS = DATA_BASE/'mtk/archipelago_waters.gpkg'
PATH_TO_GOF_WATERS = DATA_BASE/'mtk/gof_waters.gpkg'

# Hand drawn layer for fisheries
PATH_TO_FISHERIES = DATA_BASE/'fisheries_3067.gpkg'
