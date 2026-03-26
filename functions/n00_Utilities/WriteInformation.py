import datetime

def write_information(log_file, text):
    """
    Writes a message to both the console and a log file, with timestamp.

    Parameters
    ----------
    log_file : file object or str
        Either an open file handle (e.g., from open('log.txt', 'a'))
        or a string filename.
    text : str
        The message to write.
    """
    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    msg = f"{timestamp} {text}"

    if hasattr(log_file, "write"):           # open file handle
        log_file.write(msg + "\n")
        log_file.flush()                     # ensure it's written immediately
    elif isinstance(log_file, str):          # if user passed a filename
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    else:
        raise ValueError("log_file must be a file handle or filename string.")

    print(msg)
