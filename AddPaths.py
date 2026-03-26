import sys
import os

def add_paths():
    base_paths = [
        'functions/n00_Utilities',
        'functions/n00_Preprocessing',
        'functions/n01_TotalActivation',
        'functions/n02_Thresholding',
        'functions/n03_Clustering',
        'functions/n04_Regression',

        'input_scripts',
    ]

    # Add each base path and all its subdirectories
    for base_path in base_paths:
        for root, dirs, files in os.walk(base_path):
            sys.path.append(root)
