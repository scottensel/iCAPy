# main_timecourses.py
from input_scripts.Inputs_TimeCourses_data import setup_timecourses_data_params
from input_scripts.Inputs_TimeCourses import setup_timecourses_params
from functions.Run_TimeCourse import run_regression  # you will implement this

import pickle

def main():
    # Data-related params
    param = setup_timecourses_data_params()

    # Time-course regression / iCAPs TC-related params
    param.update(setup_timecourses_params())

    # Run regression to obtain iCAPs time courses
    run_regression(param)


if __name__ == "__main__":
    main()
