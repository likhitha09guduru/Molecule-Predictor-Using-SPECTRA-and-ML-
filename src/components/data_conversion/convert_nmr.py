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


def _extract_proton_peaks(data: dict, compound_id: str) -> str:
    """Extract 1H NMR peaks as a string like 'shift:intensity:multiplicity, ...'"""
    peak_data = None
    for key in ['peaks', 'PEAKS', 'Peak list', 'PEAK TABLE']:
        if key in data:
            peak_data = data[key]
            break

    if not peak_data:
        logging.warning(f"  {compound_id}: no proton peak table found")
        return ""

    proton_peaks = []
    skipped = 0
    for peak in peak_data:
        if isinstance(peak, (list, tuple)) and len(peak) >= 2:
            try:
                shift = float(peak[0])
                intensity = float(peak[1]) if len(peak) > 1 else 1.0
                mult = str(peak[2]) if len(peak) > 2 else 's'
                proton_peaks.append(f"{shift}:{intensity}:{mult}")
            except (TypeError, ValueError):
                skipped += 1
        else:
            skipped += 1

    if skipped > 0:
        logging.warning(f"  {compound_id}: skipped {skipped} malformed proton peak(s)")

    return ", ".join(proton_peaks)


def _extract_carbon_peaks(data: dict, compound_id: str) -> str:
    """Extract 13C NMR peaks as a comma-separated string of chemical shifts."""
    carbon_peaks = []
    found_key = None

    for key in ['carbon_peaks', '13C peaks', 'C13 PEAKS']:
        if key in data:
            found_key = key
            skipped = 0
            for peak in data[key]:
                if isinstance(peak, (int, float)):
                    carbon_peaks.append(str(float(peak)))
                elif isinstance(peak, (list, tuple)) and len(peak) >= 1:
                    try:
                        carbon_peaks.append(str(float(peak[0])))
                    except (TypeError, ValueError):
                        skipped += 1
                else:
                    skipped += 1
            if skipped > 0:
                logging.warning(f"  {compound_id}: skipped {skipped} malformed carbon peak(s)")
            break

    if not found_key:
        logging.info(f"  {compound_id}: no carbon (13C) peak data found (this is common - not all files include it)")

    return ", ".join(carbon_peaks)


def _extract_solvent(data: dict) -> str:
    """Extract solvent field if present, otherwise 'Unknown'."""
    for key in ['solvent', 'SOLVENT', 'Solvent']:
        if key in data:
            return str(data[key])
    return 'Unknown'


def convert_nmr_files(input_dir: str,
                      output_csv: str = "artifacts/data/converted/nmr_spectra.csv",
                      chunk_size: int = 1000):
    """
    Convert NMR files to CSV with chunking to handle large files.

    Args:
        input_dir: Folder containing .jdx or .dx files
        output_csv: Path to save the CSV file
        chunk_size: Number of spectra to buffer in memory before writing to CSV
    """
    try:
        if not HAS_JCAMP:
            logging.error("jcamp not installed. Cannot convert NMR files.")
            return None

        input_path = Path(input_dir)
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Remove any existing output so re-running doesn't create duplicate rows
        if output_path.exists():
            output_path.unlink()

        nmr_files = list(input_path.rglob("*.jdx")) + list(input_path.rglob("*.dx"))

        if not nmr_files:
            logging.error(f"No NMR files found in {input_dir}")
            return None

        logging.info(f"Found {len(nmr_files)} NMR files to convert")

        first_write = True
        buffer = []
        num_converted = 0
        num_failed = 0
        num_with_carbon = 0

        for file_path in tqdm(nmr_files, desc="Converting NMR files"):
            compound_id = Path(file_path).stem
            try:
                data = JCAMP_reader(str(file_path))

                proton_str = _extract_proton_peaks(data, compound_id)
                carbon_str = _extract_carbon_peaks(data, compound_id)
                solvent = _extract_solvent(data)

                if not proton_str and not carbon_str:
                    logging.warning(f"  Skipping {compound_id}: no usable peaks (proton or carbon) extracted")
                    num_failed += 1
                    continue

                if carbon_str:
                    num_with_carbon += 1

                buffer.append({
                    'compound_id': compound_id,
                    'proton_peaks': proton_str,
                    'carbon_peaks': carbon_str,
                    'solvent': solvent,
                    'source_file': str(file_path)
                })
                num_converted += 1

            except Exception as e:
                logging.error(f"Failed to convert {file_path.name}: {str(e)}")
                num_failed += 1
                continue

            # Flush buffer once it reaches chunk_size, keeps memory usage bounded
            if len(buffer) >= chunk_size:
                _append_to_csv(buffer, output_path, first_write)
                first_write = False
                buffer = []

        # Flush any remaining records
        if buffer:
            _append_to_csv(buffer, output_path, first_write)

        logging.info("=" * 50)
        logging.info("NMR conversion complete!")
        logging.info(f"  Converted: {num_converted}")
        logging.info(f"  With carbon (13C) data: {num_with_carbon}")
        logging.info(f"  Failed/skipped: {num_failed}")
        logging.info(f"  Data saved to {output_path}")
        logging.info("=" * 50)

        return {
            'converted': num_converted,
            'with_carbon': num_with_carbon,
            'failed': num_failed,
            'output_path': str(output_path)
        }

    except Exception as e:
        raise CustomException(e, sys)


def _append_to_csv(chunk: list, output_path: Path, first_write: bool):
    """Append a chunk of records to the CSV file."""
    df = pd.DataFrame(chunk)
    df.to_csv(output_path, mode='a', index=False, header=first_write)


if __name__ == "__main__":
    try:
        convert_nmr_files(
            input_dir="artifacts/data/raw/nmr_files/",
            output_csv="artifacts/data/converted/nmr_spectra.csv",
            chunk_size=1000
        )
    except Exception as e:
        raise CustomException(e, sys)