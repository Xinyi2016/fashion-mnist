from datetime import datetime

def now_int():
    epoch = datetime.utcfromtimestamp(0)
    return int((datetime.now() - epoch).total_seconds())
