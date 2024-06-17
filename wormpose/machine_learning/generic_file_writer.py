"""
Wrapper to save the training data to different file formats
"""

import h5py
import csv
import numpy as np
import tensorflow as tf

class GenericFileWriter(object):
    """
    Write data to different file formats depending on the open_file and write_file functions
    """

    def __init__(self, open_file=None, write_file=None):
        self.open_file = open_file
        self.write_file = write_file

    def __enter__(self):
        self.f = self.open_file()
        self.f.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.f.__exit__(exc_type, exc_val, exc_tb)

    def write(self, data):
        self.write_file(self.f, data)

# Example usage:
if __name__ == "__main__":
    # Example of using GenericFileWriter with different file formats
    # For example, writing to a CSV file
    def open_csv_file():
        return open('data.csv', 'w', newline='')

    def write_to_csv(file, data):
        csv_writer = csv.writer(file)
        csv_writer.writerow(data)

    with GenericFileWriter(open_file=open_csv_file, write_file=write_to_csv) as writer:
        writer.write([1, 2, 3, 4, 5])

    # Example of using GenericFileWriter with HDF5 file
    def open_hdf5_file():
        return h5py.File('data.h5', 'w')

    def write_to_hdf5(file, data):
        dataset = file.create_dataset('data', data=np.array(data))

    with GenericFileWriter(open_file=open_hdf5_file, write_file=write_to_hdf5) as writer:
        writer.write([1, 2, 3, 4, 5])

