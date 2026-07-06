"""
convert_nmr.py - Converts raw NMR files (.jdx, .dx) to standardized CSV format
with chunking to handle large files.
Uses the jcamp library to read JCAMP-DX files and extract relevant information.
"""
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from src.logger import logging
from src.exception import CustomException

try:
    from jcamp import JCAMP_reader
    HAS_JCAMP = True
    logging.info("jcamp loaded successfully")
except ImportError:
    HAS_JCAMP = False
    logging.warning("jcamp not installed. Install: pip install jcamp")

def convert_nmr_files(input_dir: str,
                      output_csv: str = "artifacts/data/converted/nmr_spectra.csv",
                      chunk_size: int = 1000):
    """
    Convert NMR files to CSV with chunking to handle large files.
    
    Args:
        input_dir: Folder containing .jdx or .dx files
        output_csv: Path to save the CSV file
        chunk_size: Number of spectra to process before writing to CSV
    """
    try:
        if not HAS_JCAMP:
            logging.error("jcamp not installed. Cannot convert NMR files.")
            return None

        input_path = Path(input_dir)
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Collect JCAMP files
        nmr_files = list(input_path.rglob("*.jdx")) + list(input_path.rglob("*.dx"))

        if not nmr_files:
            logging.error(f"No NMR files found in {input_dir}")
            return None

        logging.info(f"Found {len(nmr_files)} NMR files to convert")

        # Write headers once
        first_file = True

        for file_path in tqdm(nmr_files, desc="Converting NMR files"):
            try:
                data = JCAMP_reader(str(file_path))
                compound_id = Path(file_path).stem

                # --- Extract Proton (1H) NMR Peaks ---
                proton_peaks = []
                peak_data = None

                # Try different possible field names
                for key in ['peaks', 'PEAKS', 'Peak list', 'PEAK TABLE']:
                    if key in data:
                        peak_data = data[key]
                        break

                if peak_data:
                    for peak in peak_data:
                        if isinstance(peak, (list, tuple)) and len(peak) >= 2:
                            try:
                                shift = float(peak[0])
                                intensity = float(peak[1]) if len(peak) > 1 else 1.0
                                mult = str(peak[2]) if len(peak) > 2 else 's'
                                proton_peaks.append(f"{shift}:{intensity}:{mult}")
                            except:
                                pass

                proton_str = ", ".join(proton_peaks)

                # --- Extract Carbon (13C) NMR Peaks ---
                carbon_peaks = []
                for key in ['carbon_peaks', '13C peaks', 'C13 PEAKS']:
                    if key in data:
                        for peak in data[key]:
                            if isinstance(peak, (int, float)):
                                carbon_peaks.append(str(float(peak)))
                            elif isinstance(peak, (list, tuple)) and len(peak) >= 1:
                                carbon_peaks.append(str(float(peak[0])))
                        break

                carbon_str = ", ".join(carbon_peaks)

                # Extract solvent
                solvent = 'Unknown'
                for key in ['solvent', 'SOLVENT', 'Solvent']:
                    if key in data:
                        solvent = str(data[key])
                        break

                # Store in chunk
                chunk = [{
                    'compound_id': compound_id,
                    'proton_peaks': proton_str,
                    'carbon_peaks': carbon_str,
                    'solvent': solvent,
                    'source_file': str(file_path)
                }]

                # Write immediately (NMR files typically have 1 spectrum per file)
                _append_to_csv(chunk, output_path, first_file)
                first_file = False

            except Exception as e:
                logging.error(f"Failed to convert {file_path}: {str(e)}")
                continue

        logging.info(f"Conversion complete! Data saved to {output_path}")

    except Exception as e:
        raise CustomException(e, sys)

def _append_to_csv(chunk: list, output_path: Path, first_file: bool):
    """Append a chunk of data to the CSV file."""
    df = pd.DataFrame(chunk)
    df.to_csv(output_path, mode='a', index=False, header=first_file)

if __name__ == "__main__":
    try:
        convert_nmr_files(
            input_dir="artifacts/data/raw/nmr_files/",
            output_csv="artifacts/data/converted/nmr_spectra.csv",
            chunk_size=1000
        )
    except Exception as e:
        raise CustomException(e, sys)