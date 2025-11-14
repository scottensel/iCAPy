import sys
import os

def add_paths():
    base_paths = [
        'functions/Utilities',
        'functions/00_Preprocessing',
        'functions/TotalActivation',
        'functions/02_Thresholding',
        'functions/03_Clustering',
        'functions/04_Regression',

        'input_scripts',
    ]

    # Add each base path and all its subdirectories
    for base_path in base_paths:
        for root, dirs, files in os.walk(base_path):
            sys.path.append(root)
