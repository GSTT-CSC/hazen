from datetime import datetime
from hashlib import md5
from time import time
import inspect

import jwt
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy.ext.hybrid import hybrid_property
from flask_login import UserMixin
from flask import current_app

from app import db, login
from app.database import Model, SurrogatePK, CreatedTimestampMixin, UUID, JSONB


@login.user_loader
def load_user(id):
    return User.query.get(str(id))


class User(UserMixin, Model, SurrogatePK, CreatedTimestampMixin):

    def __init__(self, **kwargs):
        db.Model.__init__(self, **kwargs)

    username = db.Column(db.String(64), index=True, unique=True)
    email = db.Column(db.String(120), index=True, unique=True)
    password_hash = db.Column(db.String(128))

    acquisitions = db.relationship('Acquisition', backref='author', lazy='dynamic')

    about_me = db.Column(db.String(140))
    last_seen = db.Column(db.DateTime, default=datetime.utcnow)

    @hybrid_property
    def filesystem_key(self):
        return self.id.hex

    def avatar(self, size):
        digest = md5(self.email.lower().encode('utf-8')).hexdigest()
        return 'https://www.gravatar.com/avatar/{}?d=identicon&s={}'.format(
            digest, size)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return '<User {}>'.format(self.username)

    def get_reset_password_token(self, expires_in=600):
        return jwt.encode(
            {'reset_password': str(self.id), 'exp': time() + expires_in},
            str(current_app.config['SECRET_KEY']), algorithm='HS256').decode('utf-8')

    @staticmethod
    def verify_reset_password_token(token):
        try:
            id = jwt.decode(token, current_app.config['SECRET_KEY'],
                            algorithms=['HS256'])['reset_password']
        except:
            return
        return User.query.get(id)


class Acquisition(Model, SurrogatePK, CreatedTimestampMixin):

    def __init__(self, **kwargs):
        db.Model.__init__(self, **kwargs)
    series_instance_uid = db.Column(db.String(140))
    description = db.Column(db.String(200))
    files = db.Column(db.Integer)
    user_id = db.Column(db.ForeignKey('user.id'))

    def __repr__(self):
        return '<Acquistion {}>'.format(self.description)

    @hybrid_property
    def filesystem_key(self):
        return self.id.hex


class Fact(Model, SurrogatePK, CreatedTimestampMixin):

    def __init__(self, **kwargs):
        db.Model.__init__(self, **kwargs)

    user_id = db.Column(db.ForeignKey('user.id'))
    acquisition_id = db.Column(db.ForeignKey('acquisition.id'))
    process_task = db.Column(db.ForeignKey('process_task.id'))
    process_task_variables = db.Column(JSONB)
    data = db.Column(JSONB)
    status = db.Column(db.String)
    task = db.relationship("ProcessTask", backref="facts")


class ProcessTask(Model, SurrogatePK, CreatedTimestampMixin):
    def __init__(self, **kwargs):
        db.Model.__init__(self, **kwargs)

    name = db.Column(db.String)
    signature = db.Column(db.String)
    docstring = db.Column(db.String)
