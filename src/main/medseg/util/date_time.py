from datetime import datetime


def get_current_date_time_str():
    now = datetime.now()
    formatted_date = now.strftime("%Y-%m-%d_%H-%M-%S")
    return formatted_date
