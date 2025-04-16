import gzip
import shutil
import sys
from pathlib import Path
import os


PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
os.environ['PROJECT_ROOT'] = PROJECT_ROOT
input_path = str(Path(PROJECT_ROOT) / "data/raw/sub-01/ses-forrestgump/func/sub-01_ses-forrestgump_task-forrestgump_run-01_physio.tsv.gz")
output_path = str(Path(PROJECT_ROOT) / "data/physio/sub-01/sub-01_ses-forrestgump_task-forrestgump_run-02_physio.tsv")

with gzip.open(input_path, 'rb') as f_in:
    with open(output_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

print(f"Extracted: {output_path}")
