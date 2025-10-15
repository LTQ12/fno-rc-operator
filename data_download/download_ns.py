import os
import requests
from tqdm import tqdm

def download_pdebench_data():
    """
    Downloads the 2D Navier-Stokes dataset from the PDEBench repository
    using the requests library to handle potential redirects and large files.
    """
    print("Creating data directory if it doesn't exist...")
    # Go up one level from data_download to the project root, then into data/
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    filename = "2D_NS_incom_inhom_vsmall_v1000_t200_N1000.h5"
    # The actual download URL requires a persistentId for authorization
    base_url = "https://darus.uni-stuttgart.de/api/access/datafile"
    file_id = "248909"
    persistent_id = "doi:10.18419/darus-2986"
    url = f"{base_url}/{file_id}?persistentId={persistent_id}"
    output_path = os.path.join(data_dir, filename)
    
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000: # Check for a real file, not a placeholder
        print(f"Dataset '{filename}' already exists. Skipping download.")
        return

    print(f"Downloading {filename} from PDEBench repository...")
    print(f"URL: {url}")
    print(f"Saving to: {output_path}")

    try:
        with requests.get(url, stream=True, allow_redirects=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f, tqdm(
                total=total_size, unit='iB', unit_scale=True, desc=filename
            ) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        print("Download completed successfully!")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred during download: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    download_pdebench_data() 