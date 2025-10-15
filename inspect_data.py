import scipy.io
import numpy as np
import os

def inspect_mat_file(filepath):
    """
    Loads a .mat file and prints the keys and shapes of the data it contains.
    """
    print(f"Inspecting file: {filepath}")
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return
        
    try:
        data = scipy.io.loadmat(filepath)
        print("File loaded successfully. Contents:")
        print("-" * 30)
        for key, value in data.items():
            # The .mat file format saves extra metadata like __header__, __version__, __globals__
            # We only care about the actual numpy arrays.
            if isinstance(value, np.ndarray):
                print(f"Key: '{key}'")
                print(f"  - Shape: {value.shape}")
                print(f"  - Data type: {value.dtype}")
        print("-" * 30)

    except Exception as e:
        print(f"An error occurred while reading the file: {e}")

if __name__ == "__main__":
    # Assuming the script is run from the project root
    data_file_path = os.path.join('data', 'ns_data.mat')
    inspect_mat_file(data_file_path) 