import datetime


def write_information(log_file, text):
    """
    Writes a given text to a log file and displays it on the terminal,
    with a timestamp prepended to each message.

    Inputs:
        log_file - open file handle (e.g. from open('log.txt', 'a')) or
                   a string filename pointing to the log file to append to
        text     - str, the message to write

    Outputs:
        Writes timestamp + text to the log file and prints to the terminal.
    """
    # Create timestamp in the format [YYYY-MM-DD HH:MM:SS]
    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    msg = f"{timestamp} {text}"

    if hasattr(log_file, "write"):          # open file handle
        # Write to log file and flush immediately so nothing is lost
        # if the process crashes
        log_file.write(msg + "\n")
        log_file.flush()
    elif isinstance(log_file, str):         # filename string passed directly
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    else:
        raise ValueError("log_file must be a file handle or filename string.")

    # Display on terminal
    print(msg)
