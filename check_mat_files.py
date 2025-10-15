import scipy.io
import os
import h5py

def check_file(path):
    print(f"--- Checking file: {path} ---")
    if not os.path.exists(path):
        print("Result: File does not exist.")
        return

    try:
        # Use h5py for v7.3 files
        with h5py.File(path, 'r') as data:
            print("Result: Success! File is readable with h5py.")
            print(f"Keys inside: {list(data.keys())}")
    except Exception as e:
        print(f"Result: Failed to read file with h5py.")
        print(f"Error: {e}")
    print("-" * (25 + len(path)))
    print()


print("Starting integrity check of 'burgers.mat'...")
print("="*40)
print()

check_file('burgers.mat')

print("="*40)
print("Integrity check finished.")

print("Starting integrity check of 'burgers2.mat'...")
print("="*40)
print()

check_file('burgers2.mat')

print("="*40)
print("Integrity check finished.") 