# main_timecourses.py

from input_scripts.Inputs_TA_data import setup_data_params
from input_scripts.Inputs_TimeCourses import setup_timecourses_params
from functions.Run_TimeCourses import run_timecourses  # you will implement this


def main():
    # Data-related params
    param = setup_data_params()

    # Time-course regression / iCAPs TC-related params
    param.update(setup_timecourses_params())

    # Run regression to obtain iCAPs time courses
    run_timecourses(param)


if __name__ == "__main__":
    main()
