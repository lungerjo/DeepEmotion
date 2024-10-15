import nibabel as nib
import sys
from pathlib import Path
import os

file_path = Path("/Users/joshualunger/DeepEmotion/data/raw/sub-14/ses-forrestgump/func/sub-14_ses-forrestgump_task-forrestgump_rec-dico7Tad2grpbold7TadBrainMaskNLBrainMask_run-02_bold.nii.gzz")
print(file_path.exists())  # Should print True if the path is correct

"""
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
os.environ['PROJECT_ROOT'] = PROJECT_ROOT
print(PROJECT_ROOT)
file_path = str(Path(PROJECT_ROOT) / "data" / "raw" / "sub-04" / "ses-forrestgump" / "func" / "sub-04_ses-forrestgump_task-forrestgump_rec-dico7Tad2grpbold7TadNL_run-04_bold.nii.gz")
def get_nii_shape():

    img = nib.load(file_path)
    shape = img.get_fdata().shape
    
    print(f"Shape of '{file_path}': {shape}")
    return shape

if __name__ == "__main__":
    get_nii_shape()
"""