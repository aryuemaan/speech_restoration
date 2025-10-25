# Speech Restoration — Submission Package

**Author:** Sample Drop  
**Contents:** restore.py, report.md, README.md

## Requirements
- Python 3.8+
- NumPy
- SciPy

Install dependencies:
```bash
pip install numpy scipy
```

## Files
- `restore.py` : Main restoration script (see help `--help` for options)
- `report.md`  : 2-page technical report describing methods, parameters, and results
- `README.md`  : This file

## Quick run
Place `corrupted.wav` (and optionally `clean.wav`) in the same folder or provide full paths:
```bash
python restore.py --input ./corrupted.wav --output ./reconstructed.wav --clean ./clean.wav
```

The script will write `reconstructed.wav` and print SNR metrics if `clean.wav` is provided.

## Notes
- This implementation uses classical DSP techniques only (no ML).
- Parameters can be adjusted through CLI flags (see `--help`).
- For submission, include all three files and the WAV files in one compressed folder.

Good luck!
