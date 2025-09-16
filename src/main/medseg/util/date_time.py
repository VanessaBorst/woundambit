# --------------------------------------------------------
# WoundAmbit
# Copyright (c) 2025 Vanessa Borst, Timo Dittus and Contributors.
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from datetime import datetime


def get_current_date_time_str():
    now = datetime.now()
    formatted_date = now.strftime("%Y-%m-%d_%H-%M-%S")
    return formatted_date
