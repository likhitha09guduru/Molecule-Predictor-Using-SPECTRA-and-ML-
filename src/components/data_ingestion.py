import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Reads the fused multimodal dataset built by
            # notebook/build_multimodal_dataset.py from the REAL, published
            # "IR-NMR Multimodal Computational Spectra Dataset for 177K
            # Patent-Extracted Organic Molecules" (Zipoli, Alberts, Laino;
            # IBM Research; Zenodo record 16417648; CDLA-Permissive-2.0).
            # Run that script first (requires internet + pyarrow + rdkit)
            # to produce notebook/data/multimodal_spectra_dataset.csv.
            data_path = 'notebook/data/multimodal_spectra_dataset.csv'
            if not os.path.exists(data_path):
                raise FileNotFoundError(
                    f"{data_path} not found. Run "
                    "`python notebook/build_multimodal_dataset.py` first to "
                    "download and fuse the real IR + NMR dataset."
                )
            df = pd.read_csv(data_path)
            logging.info('Read the real, fused IR + NMR multimodal dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))
