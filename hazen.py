import os

from app import db, create_app
from app.models import User, Acquisition

__version__ = 'dev-0.1.0'
__author__ = "mohammad_haris.shuaib@kcl.ac.uk"


app = create_app()

@app.shell_context_processor
def make_shell_context():
    return {'db': db, 'User': User, 'Acquisition': Acquisition}


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=True)