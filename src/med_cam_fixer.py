import os
import site
import shutil
import sys

def fix_medcam():
    """
    Fix the numpy float type issue in medcam package.
    """
    try:
        # Find medcam_utils.py in site-packages
        medcam_utils_path = os.path.join(site.getsitepackages()[0], 'medcam/medcam_utils.py')
        print(f"Found medcam_utils.py at: {medcam_utils_path}")

        # Create backup
        backup_path = medcam_utils_path + '.backup'
        if not os.path.exists(backup_path):
            shutil.copy2(medcam_utils_path, backup_path)
            print(f"Created backup at: {backup_path}")

        # Read the file
        with open(medcam_utils_path, 'r') as f:
            lines = f.readlines()

        # Track if we made any changes
        changes_made = False

        # Fix all instances of problematic numpy float usage
        for i, line in enumerate(lines):
            if 'np.float' in line and not 'np.float64' in line:
                old_line = lines[i]
                lines[i] = lines[i].replace('np.float', 'np.float64')
                print(f"Fixed line {i}:")
                print(f"  Old: {old_line.strip()}")
                print(f"  New: {lines[i].strip()}")
                changes_made = True

        if not changes_made:
            print("No fixes needed - file already using np.float64")
            return

        # Write the modified content back
        with open(medcam_utils_path, 'w') as f:
            f.writelines(lines)

        print("\nSuccessfully fixed medcam_utils.py")
        print("You can now use medcam without the float type error")

    except Exception as e:
        print(f"Error fixing medcam: {str(e)}")
        print("If you need to restore from backup, use:")
        print(f"cp {backup_path} {medcam_utils_path}")
        sys.exit(1)

if __name__ == "__main__":
    fix_medcam()