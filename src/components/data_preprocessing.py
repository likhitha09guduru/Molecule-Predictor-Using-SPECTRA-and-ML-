"""
data_preprocessing.py - Preprocesses spectral data for deep learning.

PURPOSE: Cleans, normalizes, and standardizes spectral data from CSV files
into a format ready for PyTorch models.

WHAT THIS DOES:
1. Removes noise (peaks below threshold)
2. Normalizes peak intensities (0-1 range)
3. Sorts peaks by intensity (descending)
4. Pads/truncates to fixed length for batching
5. Encodes NMR multiplicity (s→0, d→1, t→2, q→3, etc.)
6. Handles large datasets with chunking

INPUT: Output from data_ingestion.py (dictionaries with spectra data)
OUTPUT: Preprocessed numpy arrays ready for PyTorch
"""
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import gc

from src.logger import logging
from src.exception import CustomException

# ============================================================================
# IMPORT UTILITY FUNCTIONS
# ============================================================================

from src.utils import (
    encode_multiplicity,
    validate_peaks,
    ensure_directory,
    get_peak_statistics
)


class SpectralPreprocessor:
    """
    Preprocess spectral data for deep learning models.
    
    HOW TO USE:
    -----------
    preprocessor = SpectralPreprocessor(
        max_peaks_ms=200,
        max_peaks_nmr=150,
        max_peaks_ir=100
    )
    processed_data = preprocessor.create_dataset(ingestion_data)
    """
    
    def __init__(self, 
                 max_peaks_ms: int = 200,
                 max_peaks_nmr: int = 150,
                 max_peaks_ir: int = 100,
                 noise_threshold_ms: float = 0.01,
                 noise_threshold_nmr: float = 0.001,
                 noise_threshold_ir: float = 0.01,
                 normalize: bool = True,
                 pad_value: float = 0.0,
                 chunk_size: int = 5000):
        """
        Initialize the preprocessor.
        
        Args:
            max_peaks_ms: Max number of MS peaks to keep (pads/truncates)
            max_peaks_nmr: Max number of NMR peaks to keep
            max_peaks_ir: Max number of IR peaks to keep
            noise_threshold_ms: Minimum intensity for MS peaks
            noise_threshold_nmr: Minimum intensity for NMR peaks
            noise_threshold_ir: Minimum intensity for IR peaks
            normalize: Whether to normalize intensities to 0-1
            pad_value: Value to use for padding
            chunk_size: Number of compounds to process at once (memory management)
        """
        try:
            self.max_peaks_ms = max_peaks_ms
            self.max_peaks_nmr = max_peaks_nmr
            self.max_peaks_ir = max_peaks_ir
            self.noise_threshold_ms = noise_threshold_ms
            self.noise_threshold_nmr = noise_threshold_nmr
            self.noise_threshold_ir = noise_threshold_ir
            self.normalize = normalize
            self.pad_value = pad_value
            self.chunk_size = chunk_size
            
            # Scalers for feature standardization
            self.ms_scaler = StandardScaler()
            self.nmr_scaler = StandardScaler()
            self.ir_scaler = StandardScaler()
            
            # Track if scalers are fitted
            self._scalers_fitted = False
            
            logging.info("=" * 60)
            logging.info("SPECTRAL PREPROCESSOR INITIALIZED")
            logging.info("=" * 60)
            logging.info(f"  MS: max_peaks={max_peaks_ms}, threshold={noise_threshold_ms}")
            logging.info(f"  NMR: max_peaks={max_peaks_nmr}, threshold={noise_threshold_nmr}")
            logging.info(f"  IR: max_peaks={max_peaks_ir}, threshold={noise_threshold_ir}")
            logging.info(f"  Normalize: {normalize}")
            logging.info(f"  Chunk size: {chunk_size}")
            logging.info("=" * 60)
            
        except Exception as e:
            raise CustomException(e, sys)
    
    # =========================================================================
    # PEAK PROCESSING (MS and IR)
    # =========================================================================
    
    def process_peaks(self, 
                      peaks: List[Tuple[float, float]], 
                      modality: str) -> np.ndarray:
        """
        Process peaks for a single spectrum (MS or IR).
        
        Steps:
        1. Remove noise below threshold
        2. Sort by intensity (descending)
        3. Keep top N peaks
        4. Pad or truncate to fixed length
        5. Normalize intensities
        
        Args:
            peaks: List of (value, intensity) tuples
            modality: 'ms' or 'ir'
        
        Returns:
            2D array: [[value1, intensity1], [value2, intensity2], ...]
            Shape: (max_peaks, 2)
        """
        try:
            if not peaks:
                return self._create_padded_array(modality)
            
            # Validate peaks using utility function
            if not validate_peaks(peaks):
                logging.warning(f"Invalid peaks format for {modality}, returning padded array")
                return self._create_padded_array(modality)
            
            # Get thresholds and limits
            if modality == 'ms':
                threshold = self.noise_threshold_ms
                max_peaks = self.max_peaks_ms
            else:  # ir
                threshold = self.noise_threshold_ir
                max_peaks = self.max_peaks_ir
            
            # Separate values and intensities
            values = np.array([p[0] for p in peaks if p[1] > threshold])
            intensities = np.array([p[1] for p in peaks if p[1] > threshold])
            
            if len(values) == 0:
                return self._create_padded_array(modality)
            
            # Normalize intensities to 0-1
            if self.normalize:
                max_intensity = np.max(intensities)
                if max_intensity > 0:
                    intensities = intensities / max_intensity
            
            # Sort by intensity (descending) and keep top N
            sorted_indices = np.argsort(intensities)[::-1]
            values = values[sorted_indices[:max_peaks]]
            intensities = intensities[sorted_indices[:max_peaks]]
            
            # Pad or truncate
            if len(values) < max_peaks:
                # Pad with zeros
                values = np.pad(values, (0, max_peaks - len(values)), constant_values=self.pad_value)
                intensities = np.pad(intensities, (0, max_peaks - len(intensities)), constant_values=self.pad_value)
            
            # Stack into (max_peaks, 2) array
            processed = np.column_stack([values, intensities])
            
            return processed
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def _create_padded_array(self, modality: str) -> np.ndarray:
        """Create a padded array when no peaks exist."""
        if modality == 'ms':
            max_peaks = self.max_peaks_ms
        else:  # ir
            max_peaks = self.max_peaks_ir
        
        return np.full((max_peaks, 2), self.pad_value, dtype=np.float32)
    
    # =========================================================================
    # NMR PEAK PROCESSING (with multiplicity)
    # =========================================================================
    
    def process_nmr_peaks(self, 
                          peaks: List[Tuple[float, str, float]]) -> np.ndarray:
        """
        Process NMR peaks with multiplicity information.
        
        NMR peaks have format: (chemical_shift, multiplicity, integral)
        
        Multiplicity encoding (using utils.encode_multiplicity):
            s (singlet) = 0
            d (doublet) = 1
            t (triplet) = 2
            q (quartet) = 3
            m (multiplet) = 4
            dd (doublet of doublets) = 5
            dt (doublet of triplets) = 6
            td (triplet of doublets) = 7
            dq (doublet of quartets) = 8
            qd (quartet of doublets) = 9
            brs (broad singlet) = 10
            br (broad) = 11
        
        Args:
            peaks: List of (chemical_shift, multiplicity, integral) tuples
        
        Returns:
            2D array: [[shift1, mult_code1, intensity1], ...]
            Shape: (max_peaks_nmr, 3)
        """
        try:
            if not peaks:
                return self._create_padded_nmr_array()
            
            # Filter noise and encode using utility function
            filtered = []
            for shift, mult, intensity in peaks:
                if intensity > self.noise_threshold_nmr:
                    # Using encode_multiplicity from utils.py
                    mult_code = encode_multiplicity(mult)
                    filtered.append((float(shift), int(mult_code), float(intensity)))
            
            if not filtered:
                return self._create_padded_nmr_array()
            
            # Convert to arrays
            shifts = np.array([p[0] for p in filtered])
            mult_codes = np.array([p[1] for p in filtered])
            intensities = np.array([p[2] for p in filtered])
            
            # Normalize intensities
            if self.normalize and len(intensities) > 0:
                max_int = np.max(intensities)
                if max_int > 0:
                    intensities = intensities / max_int
            
            # Sort by intensity (descending)
            sorted_idx = np.argsort(intensities)[::-1]
            shifts = shifts[sorted_idx]
            mult_codes = mult_codes[sorted_idx]
            intensities = intensities[sorted_idx]
            
            # Pad/truncate
            max_peaks = self.max_peaks_nmr
            if len(shifts) < max_peaks:
                shifts = np.pad(shifts, (0, max_peaks - len(shifts)), constant_values=self.pad_value)
                mult_codes = np.pad(mult_codes, (0, max_peaks - len(mult_codes)), constant_values=0)
                intensities = np.pad(intensities, (0, max_peaks - len(intensities)), constant_values=self.pad_value)
            else:
                shifts = shifts[:max_peaks]
                mult_codes = mult_codes[:max_peaks]
                intensities = intensities[:max_peaks]
            
            # Stack into (max_peaks, 3) array: [shift, multiplicity_code, intensity]
            processed = np.column_stack([shifts, mult_codes, intensities])
            
            return processed
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def _create_padded_nmr_array(self) -> np.ndarray:
        """Create a padded array when no NMR peaks exist."""
        return np.full((self.max_peaks_nmr, 3), self.pad_value, dtype=np.float32)
    
    # =========================================================================
    # DATASET CREATION (WITH CHUNKING)
    # =========================================================================
    
    def create_dataset(self, 
                       ingestion_data: Dict,
                       use_all_modalities: bool = True) -> Dict:
        """
        Create preprocessed datasets for training.
        
        Args:
            ingestion_data: Output from DataIngestion.load_all()
            use_all_modalities: If True, only use compounds with all 3 modalities
        
        Returns:
            Dictionary with 'train', 'val', 'test' processed data
        """
        try:
            logging.info("\n" + "=" * 60)
            logging.info("CREATING PREPROCESSED DATASETS")
            logging.info("=" * 60)
            
            ms_data = ingestion_data.get('ms', {})
            nmr_data = ingestion_data.get('nmr', {})
            ir_data = ingestion_data.get('ir', {})
            compounds = ingestion_data.get('compounds', {})
            splits = ingestion_data.get('splits', {})
            multi_modal = ingestion_data.get('multi_modal_compounds', set())
            
            # Determine which compounds to use
            if use_all_modalities:
                compound_ids = set(ms_data.keys()) & set(nmr_data.keys()) & set(ir_data.keys())
                logging.info(f"Using {len(compound_ids)} compounds with all 3 modalities")
            else:
                compound_ids = set(compounds.keys())
                logging.info(f"Using all {len(compound_ids)} compounds")
            
            # Process each split with chunking
            processed_data = {}
            
            for split_name, split_ids in splits.items():
                split_compounds = [cid for cid in split_ids if cid in compound_ids]
                
                if not split_compounds:
                    logging.warning(f"No compounds in {split_name} split")
                    processed_data[split_name] = None
                    continue
                
                logging.info(f"\nProcessing {split_name} split: {len(split_compounds)} compounds")
                
                # Process in chunks to manage memory
                X_ms_list = []
                X_nmr_list = []
                X_ir_list = []
                
                for i in range(0, len(split_compounds), self.chunk_size):
                    chunk_compounds = split_compounds[i:i + self.chunk_size]
                    logging.info(f"  Processing chunk {i//self.chunk_size + 1}: {len(chunk_compounds)} compounds")
                    
                    chunk_ms = []
                    chunk_nmr = []
                    chunk_ir = []
                    
                    for cid in chunk_compounds:
                        # Get spectra
                        ms_spec = ms_data.get(cid, {'peaks': []})
                        nmr_spec = nmr_data.get(cid, {'1H': []})
                        ir_spec = ir_data.get(cid, {'peaks': []})
                        
                        # Process MS peaks
                        ms_processed = self.process_peaks(ms_spec.get('peaks', []), 'ms')
                        chunk_ms.append(ms_processed)
                        
                        # Process NMR peaks
                        nmr_processed = self.process_nmr_peaks(nmr_spec.get('1H', []))
                        chunk_nmr.append(nmr_processed)
                        
                        # Process IR peaks
                        ir_processed = self.process_peaks(ir_spec.get('peaks', []), 'ir')
                        chunk_ir.append(ir_processed)
                    
                    # Convert chunk to numpy arrays
                    X_ms_list.append(np.array(chunk_ms, dtype=np.float32))
                    X_nmr_list.append(np.array(chunk_nmr, dtype=np.float32))
                    X_ir_list.append(np.array(chunk_ir, dtype=np.float32))
                    
                    # Free memory after each chunk
                    gc.collect()
                
                # Concatenate all chunks
                X_ms = np.concatenate(X_ms_list, axis=0) if X_ms_list else np.array([])
                X_nmr = np.concatenate(X_nmr_list, axis=0) if X_nmr_list else np.array([])
                X_ir = np.concatenate(X_ir_list, axis=0) if X_ir_list else np.array([])
                
                processed_data[split_name] = {
                    'ms': X_ms,
                    'nmr': X_nmr,
                    'ir': X_ir,
                    'compound_ids': split_compounds
                }
                
                logging.info(f"  {split_name}: {len(split_compounds)} compounds, MS shape: {X_ms.shape}")
            
            # Fit scalers on training data
            if processed_data.get('train') is not None:
                self._fit_scalers(processed_data['train'])
                self._scalers_fitted = True
                logging.info("\n✅ Scalers fitted on training data")
            
            # Apply scaling to all splits
            for split_name in ['train', 'validation', 'test']:
                if processed_data.get(split_name) is not None:
                    processed_data[split_name] = self.transform_dataset(processed_data[split_name])
                    logging.info(f"✅ Scaling applied to {split_name}")
            
            logging.info("\n" + "=" * 60)
            logging.info("PREPROCESSING COMPLETE!")
            logging.info("=" * 60)
            
            return processed_data
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def _fit_scalers(self, train_data: Dict):
        """Fit standard scalers on training data."""
        try:
            X_ms = train_data['ms']
            X_nmr = train_data['nmr']
            X_ir = train_data['ir']
            
            # MS scaler - reshape to 2D for scaler
            if X_ms.size > 0:
                ms_flat = X_ms.reshape(X_ms.shape[0], -1)
                self.ms_scaler.fit(ms_flat)
                logging.info(f"  MS scaler fitted on {ms_flat.shape[0]} samples, {ms_flat.shape[1]} features")
            
            # NMR scaler
            if X_nmr.size > 0:
                nmr_flat = X_nmr.reshape(X_nmr.shape[0], -1)
                self.nmr_scaler.fit(nmr_flat)
                logging.info(f"  NMR scaler fitted on {nmr_flat.shape[0]} samples, {nmr_flat.shape[1]} features")
            
            # IR scaler
            if X_ir.size > 0:
                ir_flat = X_ir.reshape(X_ir.shape[0], -1)
                self.ir_scaler.fit(ir_flat)
                logging.info(f"  IR scaler fitted on {ir_flat.shape[0]} samples, {ir_flat.shape[1]} features")
                
        except Exception as e:
            raise CustomException(e, sys)
    
    def transform_dataset(self, data: Dict) -> Dict:
        """Apply scaling to a dataset."""
        try:
            if data is None:
                return None
            
            X_ms = data['ms']
            X_nmr = data['nmr']
            X_ir = data['ir']
            
            # Apply MS scaler
            if X_ms.size > 0 and self._scalers_fitted:
                ms_flat = X_ms.reshape(X_ms.shape[0], -1)
                X_ms_scaled = self.ms_scaler.transform(ms_flat).reshape(X_ms.shape)
            else:
                X_ms_scaled = X_ms
            
            # Apply NMR scaler
            if X_nmr.size > 0 and self._scalers_fitted:
                nmr_flat = X_nmr.reshape(X_nmr.shape[0], -1)
                X_nmr_scaled = self.nmr_scaler.transform(nmr_flat).reshape(X_nmr.shape)
            else:
                X_nmr_scaled = X_nmr
            
            # Apply IR scaler
            if X_ir.size > 0 and self._scalers_fitted:
                ir_flat = X_ir.reshape(X_ir.shape[0], -1)
                X_ir_scaled = self.ir_scaler.transform(ir_flat).reshape(X_ir.shape)
            else:
                X_ir_scaled = X_ir
            
            return {
                'ms': X_ms_scaled,
                'nmr': X_nmr_scaled,
                'ir': X_ir_scaled,
                'compound_ids': data['compound_ids']
            }
            
        except Exception as e:
            raise CustomException(e, sys)
    
    # =========================================================================
    # UTILITY FUNCTIONS
    # =========================================================================
    
    def get_processed_shape(self, processed_data: Dict) -> Dict:
        """Get the shape of processed data for debugging."""
        shapes = {}
        for split_name, data in processed_data.items():
            if data is not None:
                shapes[split_name] = {
                    'ms': data['ms'].shape,
                    'nmr': data['nmr'].shape,
                    'ir': data['ir'].shape,
                    'num_compounds': len(data['compound_ids'])
                }
        return shapes
    
    def save_processed_data(self, processed_data: Dict, output_dir: str = "artifacts/data/processed/"):
        """Save processed data as numpy arrays for later use."""
        try:
            output_path = Path(output_dir)
            ensure_directory(output_path)
            
            for split_name, data in processed_data.items():
                if data is None:
                    continue
                
                np.save(output_path / f"{split_name}_ms.npy", data['ms'])
                np.save(output_path / f"{split_name}_nmr.npy", data['nmr'])
                np.save(output_path / f"{split_name}_ir.npy", data['ir'])
                
                # Save compound IDs
                with open(output_path / f"{split_name}_compound_ids.txt", 'w') as f:
                    for cid in data['compound_ids']:
                        f.write(f"{cid}\n")
                
                logging.info(f"Saved {split_name} data to {output_path}")
                
        except Exception as e:
            raise CustomException(e, sys)


# ============================================================================
# TESTING (WITH REAL DATA ONLY)
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🧪 TESTING DATA PREPROCESSING")
    print("=" * 60)
    
    try:
        # Import ingestion to load real data
        from src.components.data_ingestion import DataIngestion
        
        # First, ingest real data from CSV files
        print("\n📂 Loading real data from CSVs...")
        ingestor = DataIngestion(data_dir="artifacts/data/converted")
        ingestion_data = ingestor.load_all()
        
        if not ingestion_data['compounds']:
            print("\n⚠️ No data loaded. Please run conversion scripts first:")
            print("   python data_conversion/run_all_conversions.py")
            sys.exit(1)
        
        # Then, preprocess it
        print("\n🔧 Preprocessing data...")
        preprocessor = SpectralPreprocessor(
            max_peaks_ms=200,
            max_peaks_nmr=150,
            max_peaks_ir=100,
            chunk_size=5000
        )
        
        processed_data = preprocessor.create_dataset(ingestion_data)
        
        print("\n📊 PROCESSED DATA SHAPES:")
        shapes = preprocessor.get_processed_shape(processed_data)
        for split_name, shape_info in shapes.items():
            if shape_info:
                print(f"\n{split_name.upper()}:")
                print(f"  MS: {shape_info['ms']}")
                print(f"  NMR: {shape_info['nmr']}")
                print(f"  IR: {shape_info['ir']}")
                print(f"  Compounds: {shape_info['num_compounds']}")
        
        print("\n✅ Preprocessing test complete!")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("   Please run conversion scripts first to create CSV files.")
        print("   python data_conversion/run_all_conversions.py")
    except Exception as e:
        raise CustomException(e, sys)