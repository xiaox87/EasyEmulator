"""
Generate synthetic dataset and save to disk.
Run this once to create the data files.
"""

from util.data import generate_data

if __name__ == "__main__":
    generate_data(Ncosmo=100, err_percent=0.01, save_dir='./data')