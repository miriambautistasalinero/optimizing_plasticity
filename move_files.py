import os
import shutil

# Define the source and destination directories
source_folder = 'new/saved/sim'
destination_folder = 'new/saved/mini_sim'

# Ensure the destination folder exists
os.makedirs(destination_folder, exist_ok=True)

# Loop to move the files
for i in range(40):
    filename = f'SBI_data_plasticity_round1_{i}.npz'  # Adjust the filename pattern if necessary
    source_path = os.path.join(source_folder, filename)
    destination_path = os.path.join(destination_folder, filename)
    
    # Move the file
    if os.path.exists(source_path):
        shutil.move(source_path, destination_path)
        print(f'Moved: {filename}')
    else:
        print(f'File not found: {filename}')
