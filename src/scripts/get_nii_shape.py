import nibabel as nib
import sys
from pathlib import Path
import os

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
os.environ['PROJECT_ROOT'] = PROJECT_ROOT
print(PROJECT_ROOT)
file_path = str(Path(PROJECT_ROOT) / "data/raw/derivatives/non-linear_anatomical_alignment/sub-06/ses-forrestgump/func/sub-06_ses-forrestgump_task-forrestgump_rec-dico7Tad2grpbold7TadNL_run-01_bold.nii.gz")
def get_nii_shape():

    img = nib.load(file_path)
    shape = img.get_fdata().shape
    
    print(f"Shape of '{file_path}': {shape}")
    return shape

if __name__ == "__main__":
    get_nii_shape()