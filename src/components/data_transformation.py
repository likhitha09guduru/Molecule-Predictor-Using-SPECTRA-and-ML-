import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:
    """
    Builds the fusion preprocessing pipeline that combines the two
    numeric spectral modalities (IR + NMR) with the categorical
    descriptor modality (functional group + solvent) into a single
    feature vector the downstream regressor can consume.
    """

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for the multimodal data transformation:
        it fuses the IR modality, the NMR modality and the categorical
        descriptor modality into one preprocessing ColumnTransformer.
        '''
        try:
            # Numeric modalities, engineered from the real Zenodo IR-NMR
            # dataset by notebook/build_multimodal_dataset.py:
            # IR modality: intensity integrated over standard functional-group bands
            ir_modality_columns = [
                "ir_band_ohnh_stretch_3200_3550",
                "ir_band_ch_stretch_2850_3000",
                "ir_band_carbonyl_1650_1750",
                "ir_band_aromatic_1450_1600",
                "ir_band_fingerprint_500_1500",
            ]
            # NMR modality: 1H + 13C chemical-shift spectrum summary statistics
            nmr_modality_columns = [
                "h_nmr_shift_mean",
                "h_nmr_shift_std",
                "h_nmr_shift_max",
                "h_nmr_peak_count",
                "c_nmr_shift_mean",
                "c_nmr_shift_std",
                "c_nmr_shift_max",
                "c_nmr_peak_count",
            ]
            numerical_columns = ir_modality_columns + nmr_modality_columns

            # Categorical descriptor modality: real composition flags derived
            # from each molecule's SMILES via RDKit
            categorical_columns = [
                "contains_nitrogen",
                "contains_oxygen",
                "contains_halogen",
                "contains_sulfur",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())

                ]
            )

            cat_pipeline = Pipeline(

                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical (descriptor modality) columns: {categorical_columns}")
            logging.info(f"Numerical (IR + NMR modality) columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipelines", cat_pipeline, categorical_columns)

                ]

            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test multimodal data completed")

            logging.info("Obtaining preprocessing (multimodal fusion) object")

            preprocessing_obj = self.get_data_transformer_object()

            # Real target: molecular weight (g/mol), computed with RDKit from
            # each molecule's actual SMILES structure - the task is to
            # predict it purely from the fused IR + NMR spectral features.
            target_column_name = "molecular_weight"

            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                "Applying multimodal fusion preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
