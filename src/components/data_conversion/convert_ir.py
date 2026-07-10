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


def _extract_peaks(data: dict, compound_id: str) -> str:
    """
    Extract peak list from a parsed JCAMP data dict.
    Returns a peak string like "450.0:80.0, 1200.0:65.0".
    Logs (instead of silently skipping) any peak that fails to parse.
    """
    peaks = []
    peak_data = None

    for key in ['peaks', 'PEAKS', 'Peak list', 'PEAK TABLE']:
        if key in data:
            peak_data = data[key]
            break

    if not peak_data:
        logging.warning(f"  No peak table found for {compound_id}")
        return ""

    skipped = 0
    for peak in peak_data:
        if isinstance(peak, (list, tuple)) and len(peak) >= 2:
            try:
                peaks.append(f"{float(peak[0])}:{float(peak[1])}")
            except (TypeError, ValueError) as e:
                skipped += 1
        else:
            skipped += 1

    if skipped > 0:
        logging.warning(f"  {compound_id}: skipped {skipped} malformed peak(s)")

    return ", ".join(peaks)


def _extract_range(data: dict, compound_id: str) -> str:
    """
    Extract the x-axis range (wavenumber range) from a parsed JCAMP data dict.
    Falls back to a default range if missing/malformed, but logs it so the
    fallback is visible instead of silent.
    """
    default_range = "400,4000"

    if 'xrange' not in data:
        logging.warning(f"  {compound_id}: no xrange found, using default {default_range}")
        return default_range

    xrange_data = data['xrange']
    if isinstance(xrange_data, (list, tuple)) and len(xrange_data) >= 2:
        try:
            return f"{float(xrange_data[0])},{float(xrange_data[1])}"
        except (TypeError, ValueError):
            logging.warning(f"  {compound_id}: xrange present but unparseable, using default {default_range}")
            return default_range

    logging.warning(f"  {compound_id}: xrange in unexpected format, using default {default_range}")
    return default_range


def convert_ir_files(input_dir: str,
                     output_csv: str = "artifacts/data/converted/ir_spectra.csv",
                     chunk_size: int = 1000):
    """
    Convert IR files to CSV with chunking to handle large files.

    Args:
        input_dir: Folder containing .jdx or .dx files
        output_csv: Path to save the CSV file
        chunk_size: Number of spectra to buffer in memory before writing to CSV
    """
    try:
        if not HAS_JCAMP:
            logging.error("jcamp not installed. Cannot convert IR files.")
            return None

        input_path = Path(input_dir)
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Remove any existing output file so re-runs don't append to stale data
        if output_path.exists():
            output_path.unlink()

        ir_files = list(input_path.rglob("*.jdx")) + list(input_path.rglob("*.dx"))

        if not ir_files:
            logging.error(f"No IR files found in {input_dir}")
            return None

        logging.info(f"Found {len(ir_files)} IR files to convert")

        first_write = True
        buffer = []
        num_converted = 0
        num_failed = 0

        for file_path in tqdm(ir_files, desc="Converting IR files"):
            compound_id = Path(file_path).stem
            try:
                data = JCAMP_reader(str(file_path))

                peaks_str = _extract_peaks(data, compound_id)
                range_str = _extract_range(data, compound_id)

                if not peaks_str:
                    logging.warning(f"  Skipping {compound_id}: no usable peaks extracted")
                    num_failed += 1
                    continue

                buffer.append({
                    'compound_id': compound_id,
                    'peaks': peaks_str,
                    'range': range_str,
                    'source_file': str(file_path)
                })
                num_converted += 1

            except Exception as e:
                logging.error(f"Failed to convert {file_path.name}: {str(e)}")
                num_failed += 1
                continue

            # Flush buffer to disk once it reaches chunk_size,
            # so memory usage stays bounded on large datasets
            if len(buffer) >= chunk_size:
                _append_to_csv(buffer, output_path, first_write)
                first_write = False
                buffer = []

        # Flush any remaining records after the loop ends
        if buffer:
            _append_to_csv(buffer, output_path, first_write)

        logging.info("=" * 50)
        logging.info(f"Conversion complete!")
        logging.info(f"  Converted: {num_converted}")
        logging.info(f"  Failed/skipped: {num_failed}")
        logging.info(f"  Data saved to {output_path}")
        logging.info("=" * 50)

        return {'converted': num_converted, 'failed': num_failed, 'output_path': str(output_path)}

    except Exception as e:
        raise CustomException(e, sys)


def _append_to_csv(chunk: list, output_path: Path, first_write: bool):
    """Append a chunk of records to the CSV file."""
    df = pd.DataFrame(chunk)
    df.to_csv(output_path, mode='a', index=False, header=first_write)


if __name__ == "__main__":
    try:
        convert_ir_files(
            input_dir="artifacts/data/raw/ir_files/",
            output_csv="artifacts/data/converted/ir_spectra.csv",
            chunk_size=1000
        )
    except Exception as e:
        raise CustomException(e, sys)