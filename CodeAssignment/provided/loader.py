import time
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor


# Helper function to load a chunk of data from the CSV file
def load_chunk(filename, skiprows, nrows):
    """Load a chunk of data from a CSV file."""
    data_chunk = pd.read_csv(filename, skiprows=skiprows, nrows=nrows, header=None)
    return data_chunk


class SingleProcessDataset(Dataset):
    def __init__(self, csv_file):
        start_time = time.time()
        print("Loading data using single process...")
        
        self.data = pd.read_csv(csv_file)
        self.features = torch.FloatTensor(self.data[['x1', 'x2', 'x3']].values)
        self.labels = torch.LongTensor(self.data['label'].values)
        
        # Calculate total load time
        self.load_time = time.time() - start_time
        print(f"Dataset loading completed in {self.load_time:.2f} seconds")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_chunk(filename, skiprows, nrows):
    """Helper function to load a chunk of data from a CSV file."""
    data_chunk = pd.read_csv(filename, skiprows=skiprows, nrows=nrows)
    return data_chunk


class MultiProcessDataset(SingleProcessDataset):
    def __init__(self, csv_file, num_workers=4):
        start_time = time.time()
        print("Loading data using multi process...")

        # Read the first few rows to determine data size
        data = pd.read_csv(csv_file, nrows=10)
        total_rows = pd.read_csv(csv_file).shape[0]
        chunk_size = total_rows // num_workers

        # Parallel data loading
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Start loading chunks in parallel
            futures = []
            for i in range(num_workers):
                skiprows = i * chunk_size + 1  # skip header for each chunk after the first
                nrows = chunk_size if i < num_workers - 1 else total_rows - i * chunk_size
                futures.append(executor.submit(load_chunk, csv_file, skiprows, nrows))

            # Collect results from all processes
            chunks = [future.result() for future in futures]

        # Concatenate all data chunks
        self.data = pd.concat(chunks, ignore_index=True)
        self.features = torch.FloatTensor(self.data[['x1', 'x2', 'x3']].values)
        self.labels = torch.LongTensor(self.data['label'].values)

        # Calculate total load time
        self.load_time = time.time() - start_time
        print(f"Dataset loading completed in {self.load_time:.2f} seconds")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
