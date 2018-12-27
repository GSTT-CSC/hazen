import matplotlib
matplotlib.use('Agg')
import os
import signal

from flask import Flask
import pydicom

from tests import TEST_DATA_DIR
from hazen import snr

# have to hack matplotlib like this to make Heroku happy

app = Flask(__name__)

signal.signal(signal.SIGINT, lambda s, f: os._exit(0))


@app.route("/")
def show_snr():
    val = str(snr.main(pydicom.read_file(str(TEST_DATA_DIR / 'snr' / 'uniform-circle.IMA'))))
    page = '<html><body><h1>SNR is :'
    page += val
    page += '</h1></body></html>'
    return page


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)))