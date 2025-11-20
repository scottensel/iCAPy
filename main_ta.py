from input_scripts.Inputs_TA_data import setup_data_params
from input_scripts.Inputs_TA import setup_ta_params
from functions.Run_TA import run_ta

def main():

    # Initialize parameters by combining both input setups
    param = setup_data_params()

    param.update(setup_ta_params())  # Integrate TA-specific settings

    # Run Total Activation (TA) with the configured parameters
    run_ta(param)

if __name__ == "__main__":
    main()


