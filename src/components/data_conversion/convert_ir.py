"""
convert_ir.py - Converts raw IR files (.jdx, .dx) to standardized CSV format
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


def convert_ir_files(input_dir: str,
                     output_csv: str = "artifacts/data/converted/ir_spectra.csv",
                     chunk_size: int = 1000):
    """
    Convert IR files to CSV with chunking to handle large files.
    
    Args:
        input_dir: Folder containing .jdx or .dx files
        output_csv: Path to save the CSV file
        chunk_size: Number of spectra to process before writing to CSV
    """
    try:
        if not HAS_JCAMP:
            logging.error("jcamp not installed. Cannot convert IR files.")
            return None

        input_path = Path(input_dir)
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Collect IR files
        ir_files = list(input_path.rglob("*.jdx")) + list(input_path.rglob("*.dx"))

        if not ir_files:
            logging.error(f"No IR files found in {input_dir}")
            return None

        logging.info(f"Found {len(ir_files)} IR files to convert")

        # Write headers once
        first_file = True

        for file_path in tqdm(ir_files, desc="Converting IR files"):
            try:
                data = JCAMP_reader(str(file_path))
                compound_id = Path(file_path).stem

                # Extract peaks
                peaks = []
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
                                peaks.append(f"{float(peak[0])}:{float(peak[1])}")
                            except:
                                pass

                peaks_str = ", ".join(peaks)

                # Extract range (if available)
                range_str = "400,4000"
                if 'xrange' in data:
                    xrange_data = data['xrange']
                    if isinstance(xrange_data, (list, tuple)) and len(xrange_data) >= 2:
                        range_str = f"{float(xrange_data[0])},{float(xrange_data[1])}"

                # Store in chunk
                chunk = [{
                    'compound_id': compound_id,
                    'peaks': peaks_str,
                    'range': range_str,
                    'source_file': str(file_path)
                }]

                # Write immediately (IR files typically have 1 spectrum per file)
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
        convert_ir_files(
            input_dir="artifacts/data/raw/ir_files/",
            output_csv="artifacts/data/converted/ir_spectra.csv",
            chunk_size=1000
        )
    except Exception as e:
        raise CustomException(e, sys)