import os
import pandas as pd
from typing import Literal

MARS_TRAINING_PATH: str = 'space_apps_2024_seismic_detection/data/mars/training/data/'
MARS_CATALOGS_PATH: str = 'space_apps_2024_seismic_detection/data/mars/training/catalogs/'
MARS_TEST_PATH: str = 'space_apps_2024_seismic_detection/data/mars/test/data/'
MARS_CATALOG: str = 'Mars_InSight_training_catalog_final.csv'

# Lunar paths
LUNAR_TRAINING_PATH: str = 'space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA/'
LUNAR_CATALOGS_PATH: str = 'space_apps_2024_seismic_detection/data/lunar/training/catalogs'
LUNAR_TEST_PATH: str = 'space_apps_2024_seismic_detection/data/lunar/test/data/S12_GradeB/'
LUNAR_CATALOG: str = 'apollo12_catalog_GradeA_final.csv'


class Dataset:
    def __init__(self, catalog_pos: pd.Series, type_: Literal['moon', 'mars']) -> None:
        self.dataset_data = catalog_pos
        self.arrival_time = catalog_pos['time_rel(sec)']  # Este es el valor real del inicio del terremoto
        self.filename = self.dataset_data.filename
        self.type_ = type_

    def get_path(self) -> str:
        base_path = MARS_TRAINING_PATH if self.type_ == 'mars' else LUNAR_TRAINING_PATH
        return os.path.join('data', base_path, self.filename)

    def get_dataframe(self) -> pd.DataFrame:
        ext: str = '.csv' if self.type_ == 'moon' else ''
        return pd.read_csv(self.get_path() + ext)

    def get_arrival_time(self) -> float:
        return self.arrival_time  # Este valor se utilizará como etiqueta durante el entrenamiento
    
    def get_filename(self) -> str:
        return os.path.basename(self.filename)