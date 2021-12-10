import os
basedir = os.path.abspath(os.path.dirname(__file__))


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'postgresql://localhost:5432/hazen'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    MAIL_SERVER = os.environ.get('MAIL_SERVER')
    MAIL_PORT = int(os.environ.get('MAIL_PORT') or 25)
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS') is not None
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
    ADMINS = [os.environ.get('ADMIN_EMAIL')] or ['haris.shuaib@gmail.com']

    ACQUISITIONS_PER_PAGE = 9

    LANGUAGES = ['en-GB', 'fr']

    UPLOADED_PATH = os.path.join(basedir, 'uploads')
    DROPZONE_MAX_FILE_SIZE = 3
    DROPZONE_MAX_FILES = 20
    DROPZONE_UPLOAD_ON_CLICK = True
    DROPZONE_ALLOWED_FILE_TYPE = 'application/dicom, .IMA'
    DROPZONE_ALLOWED_FILE_CUSTOM = True


