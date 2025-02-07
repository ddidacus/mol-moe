import time
import os

def get_uniform_subfolders(path, N):
    """
    Get N uniformly picked subfolder paths from the given directory.

    Parameters:
    path (str): The directory path from which to list subfolders.
    N (int): The number of subfolders to return.

    Returns:
    list: A list of full paths of N uniformly picked subfolders.
    """
    try:
        # List all subdirectories in the given path
        subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
        
        # Check if there are enough subfolders
        if len(subfolders) < N:
            print(f"Warning: Only {len(subfolders)} subfolders found, returning all.")
            return [os.path.abspath(folder) for folder in subfolders]

        # Randomly select N unique subfolders
        idxs = list(range(0, len(subfolders), len(subfolders) // (N)))[:N-1]
        selected_subfolders = [subfolders[idx] for idx in idxs]
        
        # Return the full paths
        return [os.path.abspath(folder) for folder in selected_subfolders + [subfolders[-1]]]

    except FileNotFoundError:
        print(f"Error: The directory '{path}' does not exist.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def get_most_recent_subfolder(directory_path):
    most_recent_subfolder = None
    most_recent_time = 0
    # Iterate over the entries in the directory
    with os.scandir(directory_path) as entries:
        for entry in entries:
            # Check if the entry is a directory
            if entry.is_dir():
                # Get the modification time
                mod_time = entry.stat().st_mtime
                # Update if this directory is more recent
                if mod_time > most_recent_time:
                    most_recent_time = mod_time
                    most_recent_subfolder = entry.path
    return most_recent_subfolder

def count_subfolders(path):
    count = 0
    for entry in os.listdir(path):
        if os.path.isdir(os.path.join(path, entry)):
            count += 1
    return count