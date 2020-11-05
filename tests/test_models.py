from datetime import datetime, timedelta
import unittest
import time

from app import create_app, db
from config import Config
from app.models import User, Acquisition, Fact, ProcessTask
from flask import current_app
from app.database import JSONB


class TestConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'postgres://test_user:test_user_passowrd@localhost:5432/hazen_test'


class TestUserModel(unittest.TestCase):

    def setUp(self):
        self.app = create_app(TestConfig)
        self.app_context = self.app.app_context()
        self.app_context.push()
        db.create_all()

    def tearDown(self):
        db.session.remove()
        db.drop_all()
        self.app_context.pop()

    def test_password_hashing(self):
        u = User(username='susan')
        u.set_password('cat')
        self.assertFalse(u.check_password('dog'))
        self.assertTrue(u.check_password('cat'))

    def test_avatar(self):
        u = User(username='john', email='john@example.com')
        self.assertEqual(u.avatar(128), ('https://www.gravatar.com/avatar/'
                                         'd4c74594d841139328695756648b6bd6'
                                         '?d=identicon&s=128'))

    def test_get_password_reset_token(self):
        u = User(username='jess')
        current_app.config.SECRET_KEY = 'cat'
        token = u.get_reset_password_token()
        self.assertTrue(isinstance(token, str))

    def test_verify_password_reset_token(self):
        u = User(username='jess')
        current_app.config.SECRET_KEY = 'cat'
        db.session.add(u)
        db.session.commit()
        token = u.get_reset_password_token()
        self.assertTrue(isinstance(token, str))


class TestAcquistionModel(unittest.TestCase):

    def setUp(self):
        self.app = create_app(TestConfig)
        self.app_context = self.app.app_context()
        self.app_context.push()
        db.create_all()

    def tearDown(self):
        db.session.remove()
        db.drop_all()
        self.app_context.pop()

    def test_acquisition(self):
        acq = Acquisition(series_instance_uid='test', description='cat')
        self.assertEqual('<Acquistion {}>'.format(acq.description), '<Acquistion cat>')


class TestProcessTaskModelCase(unittest.TestCase):

    def setUp(self):
        self.app = create_app(TestConfig)
        self.app_context = self.app.app_context()
        self.app_context.push()
        db.create_all()

    def tearDown(self):
        db.session.remove()
        db.drop_all()
        self.app_context.pop()

    def test_processtask(self):
        proc = ProcessTask(name='jess', signature='cat', docstring='this is jess')
        self.assertEqual(proc.docstring, ('this is jess'))


class TestFactModel(unittest.TestCase):

    def setUp(self):
        self.app = create_app(TestConfig)
        self.app_context = self.app.app_context()
        self.app_context.push()
        db.create_all()

    def tearDown(self):
        db.session.remove()
        db.drop_all()
        self.app_context.pop()

    def test_fact(self):
        fact = Fact(user_id='12', acquisition_id='13', process_task='14')
        self.assertEqual(fact.user_id, '12')


if __name__ == '__main__':
    unittest.main(verbosity=2)
