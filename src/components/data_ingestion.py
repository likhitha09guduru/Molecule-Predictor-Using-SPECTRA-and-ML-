#Data Ingestion Module:Loads CSV files in chunks to prevent memory overflow.
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from pathlib import Path
from sklearn.model_selection import train_test_split
import gc
from src.logger import logging
from src.exception import CustomException
from src.utils import parse_peak_string, parse_nmr_peaks, validate_compound_id, ensure_directory

class DataIngestion:
    def __init__(self, data_dir: str = "artifacts/data/converted", chunk_size: int = 10000):
        try:
            self.data_dir = Path(data_dir)
            ensure_directory(self.data_dir)
            self.chunk_size = chunk_size
            
            # Storage dictionaries
            self.compounds = {}
            self.ms_spectra = {}
            self.nmr_spectra = {}
            self.ir_spectra = {}
            
            self.multi_modal_compounds = set()
            self.single_modal_compounds = set()
            
            logging.info("=" * 10)
            logging.info("DATA INGESTION INITIALIZED")
            logging.info("=" * 10)
            logging.info(f"Data directory: {self.data_dir}")
            logging.info(f"Chunk size: {chunk_size} rows")
            logging.info("=" * 10)
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def load_all(self, 
                 ms_csv: Optional[Union[str, Path]] = None,
                 nmr_csv: Optional[Union[str, Path]] = None,
                 ir_csv: Optional[Union[str, Path]] = None,
                 test_size: float = 0.2,
                 val_size: float = 0.15) -> Dict:
        try:
            logging.info("\n" + "=" * 10)
            logging.info("STARTING DATA INGESTION")
            logging.info("=" * 10)
            
            # Set default paths
            if ms_csv is None:
                ms_csv = self.data_dir / "ms_spectra.csv"
            if nmr_csv is None:
                nmr_csv = self.data_dir / "nmr_spectra.csv"
            if ir_csv is None:
                ir_csv = self.data_dir / "ir_spectra.csv"
            
            # Load each modality
            self._load_ms_csv_chunked(ms_csv)
            self._load_nmr_csv_chunked(nmr_csv)
            self._load_ir_csv_chunked(ir_csv)
            
            # Align compounds and create splits
            self.align_compounds()
            splits = self.create_splits(test_size=test_size, val_size=val_size)
            self.save_processed_data()
            
            logging.info("\n" + "=" * 10)
            logging.info("DATA INGESTION COMPLETE!")
            logging.info("=" * 10)
            logging.info(f"Total compounds: {len(self.compounds)}")
            logging.info(f"Multi-modal (all 3): {len(self.multi_modal_compounds)}")
            logging.info(f"Train: {len(splits['train'])} | Val: {len(splits['validation'])} | Test: {len(splits['test'])}")
            logging.info("=" * 10)
            
            return {
                'ms': self.ms_spectra,
                'nmr': self.nmr_spectra,
                'ir': self.ir_spectra,
                'compounds': self.compounds,
                'splits': splits,
                'multi_modal_compounds': self.multi_modal_compounds
            }
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def _load_ms_csv_chunked(self, file_path: Path) -> None:
        try:
            if not file_path.exists():
                logging.error(f"MS CSV not found: {file_path}")
                raise FileNotFoundError(f"Required file not found: {file_path}")
            
            logging.info(f"Loading MS data from: {file_path}")
            
            chunk_count = 0
            for chunk in pd.read_csv(file_path, chunksize=self.chunk_size):
                chunk_count += 1
                logging.info(f"Processing chunk {chunk_count} ({len(chunk)} rows)")
                
                for _, row in chunk.iterrows():
                    compound_id = str(row['compound_id'])
                    
                    if not validate_compound_id(compound_id):
                        logging.warning(f"Invalid compound ID: {compound_id}, skipping")
                        continue
                    
                    self.compounds[compound_id] = self.compounds.get(compound_id, {})
                    self.compounds[compound_id].update({
                        'smiles': row.get('smiles', ''),
                        'source': str(file_path)
                    })
                    
                    peaks = parse_peak_string(row.get('peaks', ''))
                    self.ms_spectra[compound_id] = {
                        'peaks': peaks,
                        'precursor_mz': row.get('precursor_mz'),
                        'charge': row.get('charge', 1)
                    }
                
                gc.collect()
            
            logging.info(f"Loaded {len(self.ms_spectra)} mass spectra")
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def _load_nmr_csv_chunked(self, file_path: Path) -> None:
        try:
            if not file_path.exists():
                logging.error(f"NMR CSV not found: {file_path}")
                raise FileNotFoundError(f"Required file not found: {file_path}")
            
            logging.info(f"Loading NMR data from: {file_path}")
            
            chunk_count = 0
            for chunk in pd.read_csv(file_path, chunksize=self.chunk_size):
                chunk_count += 1
                logging.info(f"Processing chunk {chunk_count} ({len(chunk)} rows)")
                
                for _, row in chunk.iterrows():
                    compound_id = str(row['compound_id'])
                    
                    if not validate_compound_id(compound_id):
                        continue
                    
                    proton_peaks = parse_nmr_peaks(row.get('proton_peaks', ''))
                    carbon_peaks = parse_nmr_peaks(row.get('carbon_peaks', ''))
                    
                    self.nmr_spectra[compound_id] = {
                        '1H': proton_peaks,
                        '13C': carbon_peaks,
                        'solvent': row.get('solvent', 'Unknown')
                    }
                    
                    self.compounds[compound_id] = self.compounds.get(compound_id, {})
                    self.compounds[compound_id]['source'] = str(file_path)
                
                gc.collect()
            
            logging.info(f"Loaded {len(self.nmr_spectra)} NMR spectra")
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def _load_ir_csv_chunked(self, file_path: Path) -> None:
        try:
            if not file_path.exists():
                logging.error(f"IR CSV not found: {file_path}")
                raise FileNotFoundError(f"Required file not found: {file_path}")
            
            logging.info(f"Loading IR data from: {file_path}")
            
            chunk_count = 0
            for chunk in pd.read_csv(file_path, chunksize=self.chunk_size):
                chunk_count += 1
                logging.info(f"Processing chunk {chunk_count} ({len(chunk)} rows)")
                
                for _, row in chunk.iterrows():
                    compound_id = str(row['compound_id'])
                    
                    if not validate_compound_id(compound_id):
                        continue
                    
                    peaks = parse_peak_string(row.get('peaks', ''))
                    
                    range_str = row.get('range', '400,4000')
                    try:
                        low, high = map(float, str(range_str).split(','))
                        ir_range = (low, high)
                    except:
                        ir_range = (400, 4000)
                    
                    self.ir_spectra[compound_id] = {
                        'peaks': peaks,
                        'range': ir_range
                    }
                    
                    self.compounds[compound_id] = self.compounds.get(compound_id, {})
                    self.compounds[compound_id]['source'] = str(file_path)
                
                gc.collect()
            
            logging.info(f"Loaded {len(self.ir_spectra)} IR spectra")
            
        except Exception as e:
            raise CustomException(e, sys)
    
    # ALIGNMENT AND SPLITS
    def align_compounds(self) -> None:
        """Align compounds across modalities."""
        logging.info("\n--- Aligning Compounds Across Modalities ---")
        
        all_ids = set(self.compounds.keys())
        ms_ids = set(self.ms_spectra.keys())
        nmr_ids = set(self.nmr_spectra.keys())
        ir_ids = set(self.ir_spectra.keys())
        
        self.multi_modal_compounds = all_ids.intersection(ms_ids, nmr_ids, ir_ids)
        self.single_modal_compounds = all_ids.intersection(ms_ids.union(nmr_ids, ir_ids))
        
        logging.info(f"Total: {len(all_ids)} | MS: {len(ms_ids)} | NMR: {len(nmr_ids)} | IR: {len(ir_ids)}")
        logging.info(f"Multi-modal (all 3): {len(self.multi_modal_compounds)}")
    
    def create_splits(self, test_size: float = 0.2, val_size: float = 0.15) -> Dict[str, List]:
        """Create train/validation/test splits."""
        logging.info("\n--- Creating Data Splits ---")
        
        compound_list = list(self.multi_modal_compounds)
        if len(compound_list) < 20:
            logging.warning(f"Only {len(compound_list)} multi-modal compounds. Using all compounds.")
            compound_list = list(self.single_modal_compounds)
        
        if len(compound_list) < 5:
            logging.error("Not enough compounds found for splitting.")
            return {'train': [], 'validation': [], 'test': []}
        
        train_val, test = train_test_split(compound_list, test_size=test_size, random_state=42)
        val_ratio = val_size / (1 - test_size)
        train, val = train_test_split(train_val, test_size=val_ratio, random_state=42)
        
        splits = {'train': train, 'validation': val, 'test': test}
        logging.info(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
        return splits
    
    def save_processed_data(self) -> None:
        """Save summary of processed data."""
        processed_dir = self.data_dir.parent / "processed"
        ensure_directory(processed_dir)
        
        summary = {
            'compound_id': list(self.compounds.keys()),
            'smiles': [self.compounds[cid].get('smiles', '') for cid in self.compounds.keys()],
            'has_ms': [cid in self.ms_spectra for cid in self.compounds.keys()],
            'has_nmr': [cid in self.nmr_spectra for cid in self.compounds.keys()],
            'has_ir': [cid in self.ir_spectra for cid in self.compounds.keys()]
        }
        pd.DataFrame(summary).to_csv(processed_dir / "compound_summary.csv", index=False)
        logging.info(f"Summary saved to {processed_dir}/compound_summary.csv")
    
    def get_dataset_summary(self) -> Dict:
        """Get summary statistics."""
        return {
            'total_compounds': len(self.compounds),
            'ms_spectra': len(self.ms_spectra),
            'nmr_spectra': len(self.nmr_spectra),
            'ir_spectra': len(self.ir_spectra),
            'multi_modal': len(self.multi_modal_compounds)
        }


if __name__ == "__main__":
    try:
        print("\n" + "=" * 60)
        print(" TESTING DATA INGESTION")
        print("=" * 60)
        
        # The test expects real CSV files to exist
        ingestor = DataIngestion(data_dir="artifacts/data/converted", chunk_size=5000)
        data = ingestor.load_all()
        
        print("\nDATA SUMMARY:")
        summary = ingestor.get_dataset_summary()
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        print("\n Data ingestion test complete!")
        
    except Exception as e:
        raise CustomException(e, sys)