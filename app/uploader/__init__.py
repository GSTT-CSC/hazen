from flask import Blueprint

bp = Blueprint('uploader', __name__, template_folder='templates')

from app.uploader import routes