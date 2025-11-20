# main_thresholding.py

from input_scripts.Inputs_TA_data import setup_data_params
from input_scripts.Inputs_Thresholding import setup_thresholding_params
from functions.Run_Thresholding import run_thresholding  # you will implement this


def main():
    # Data-related params
    param = setup_data_params()

    # Thresholding-specific params
    param.update(setup_thresholding_params())

    # Run thresholding of innovation frames
    run_thresholding(param)


if __name__ == "__main__":
    main()
