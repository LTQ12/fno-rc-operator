import gdown
import os
import h5py
import numpy as np
import zipfile

def download_and_convert_darcy_2d(file_id, outdir="data"):
    """
    Downloads the 2D Darcy Flow dataset from Google Drive, unzips it,
    and converts the .mat files to .hdf5 format.
    """
    os.makedirs(outdir, exist_ok=True)
    zip_path = os.path.join(outdir, "darcy_2d_data.zip")

    # --- 1. Download ---
    if not os.path.exists(zip_path):
        print("Downloading 2D Darcy dataset from Google Drive...")
        gdown.download(id=file_id, output=zip_path, quiet=False)
    else:
        print("Dataset zip file already exists.")

    # --- 2. Unzip ---
    print("Unzipping dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(outdir)
    print("Unzipping complete.")
    
    # --- 3. Convert to HDF5 ---
    # The zip contains piececonst_r241_N1024_smooth1.mat and piececonst_r241_N1024_smooth2.mat
    mat_files = [f for f in os.listdir(outdir) if f.endswith('.mat')]
    for mat_filename in mat_files:
        mat_path = os.path.join(outdir, mat_filename)
        hdf5_path = mat_path.replace(".mat", ".hdf5")
        
        if not os.path.exists(hdf5_path):
            print(f"Converting {mat_filename} to HDF5...")
            try:
                import scipy.io
                data = scipy.io.loadmat(mat_path)
                a = data["coeff"]
                u = data["sol"]
                with h5py.File(hdf5_path, "w") as f:
                    f.create_dataset("coeff", data=a, dtype=np.float32)
                    f.create_dataset("sol", data=u, dtype=np.float32)
                print(f"Successfully converted {mat_filename}.")
            except Exception as e:
                print(f"Could not convert {mat_filename}. Error: {e}")
        else:
            print(f"{hdf5_path} already exists. Skipping conversion.")

if __name__ == "__main__":
    # The Google Drive file ID for Darcy_241 dataset found in NVIDIA's documentation
    DARCY_FILE_ID = "1O_KNn_x02Hcbx0y2yNp32tflT2T6x7V5"
    download_and_convert_darcy_2d(DARCY_FILE_ID)
    print("\nData download and conversion process finished.") 