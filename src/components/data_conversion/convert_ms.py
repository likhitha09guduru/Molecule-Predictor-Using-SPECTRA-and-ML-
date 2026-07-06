"""convert_ms.py file converts raw mass spectroscopy files (.mzML,.mgf,.msp) 
to a standardized csv format with chunking to handle large files.
It uses the matchms library to read the raw files and extract relavant information."""

import sys 
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from src.logger import logging
from src.exception import CustomException

try:
    from matchms.importing import load_from_mzml, load_from_mgf, load_from_msp
    HAS_MATCHMS = True
    logging.info("matchms loaded successfully")
except ImportError:
    HAS_MATCHMS = False
    logging.warning("matchms not installed. Install: pip install matchms")

def convert_ms_files(input_dir: str, 
                     output_csv: str = "artifacts/data/converted/ms_spectra.csv",
                     chunk_size: int = 1000):
    """Convert MS files to CSV with chunking to handle large files.
        input_dir: it is the folder containing your mass spectrometry files like .mzml, .mgf, or .msp 
        output_csv: Path to save the CSV file
        chunk_size: Number of spectra to process before writing to CSV file"""
    try:
        if not HAS_MATCHMS:
            logging.error("matchms not installed. Cannot convert MS files.")
            return None
        
        input_path = Path(input_dir)
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
         
        # Collect all MS files
        ms_files = []
        for ext in ['*.mzml', '*.mgf', '*.msp']:
            ms_files.extend(input_path.rglob(ext))
        
        if not ms_files:
            logging.error(f"No MS files found in {input_dir}")
            return None 
        
        logging.info(f"Found {len(ms_files)} MS files to convert")
        
        # Open CSV file for writing (write headers once)
        first_file = True
        
        for file_path in tqdm(ms_files, desc="Converting MS files"):
            try:
                ext = file_path.suffix.lower()
                if ext == '.mzml':
                    spectra_generator = load_from_mzml(str(file_path))
                elif ext == '.mgf':
                    spectra_generator = load_from_mgf(str(file_path))
                elif ext == '.msp':
                    spectra_generator = load_from_msp(str(file_path))
                else:
                    continue
                
                # Process spectra in chunks
                chunk = []
                for spec in spectra_generator:
                    # Extract data from spectrum
                    compound_id = spec.metadata.get('compound_id', 
                                   spec.metadata.get('name', 
                                   spec.metadata.get('title', file_path.stem)))
                    
                    smiles = spec.metadata.get('smiles', spec.metadata.get('SMILES', ''))
                    precursor_mz = spec.metadata.get('precursor_mz', None)
                    charge = spec.metadata.get('charge', 1)
                    
                    # Convert peaks to string
                    peaks_str = ", ".join([f"{mz:.4f}:{intensity:.1f}" 
                                           for mz, intensity in zip(spec.peaks.mz, spec.peaks.intensities)])
                    
                    chunk.append({
                        'compound_id': str(compound_id),
                        'smiles': str(smiles),
                        'precursor_mz': precursor_mz,
                        'charge': charge,
                        'peaks': peaks_str,
                        'source_file': str(file_path)
                    })
                    
                    # When chunk reaches limit, write to CSV
                    if len(chunk) >= chunk_size:
                        _append_to_csv(chunk, output_path, first_file)
                        first_file = False
                        chunk = []  # Clear chunk
                        logging.info(f"Written {chunk_size} spectra to CSV")
                
                # Write remaining spectra from this file
                if chunk:
                    _append_to_csv(chunk, output_path, first_file)
                    first_file = False
                    logging.info(f"Written final {len(chunk)} spectra from {file_path.name}")
                    
            except Exception as e:
                logging.error(f"Failed to convert {file_path}: {str(e)}")
                continue
        
        logging.info(f" Conversion complete! Data saved to {output_path}")
        
    except Exception as e:
        raise CustomException(e, sys)

def _append_to_csv(chunk: list, output_path: Path, first_file: bool):
    df = pd.DataFrame(chunk)
    df.to_csv(output_path, mode='a', index=False, header=first_file)

if __name__ == "__main__":
    try:
        convert_ms_files(
            input_dir="artifacts/data/raw/ms_files/",
            output_csv="artifacts/data/converted/ms_spectra.csv",
            chunk_size=1000  # ← Process 1000 spectra at a time
        )
    except Exception as e:
        raise CustomException(e, sys)