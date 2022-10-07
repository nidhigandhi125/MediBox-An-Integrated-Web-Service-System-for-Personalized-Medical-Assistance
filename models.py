import threading

from importlib_metadata import metadata
from flask_login import UserMixin
from sqlalchemy import Column, String, Integer, ForeignKey
from sqlalchemy import MetaData, Table, String, Column, Text, DateTime, Boolean, BigInteger, Integer
from datetime import date, time
from flask_sqlalchemy import SQLAlchemy
from flask import Blueprint
from settings import db, engine
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

metadata = MetaData()

db = SQLAlchemy()

models = Blueprint('models', __name__)


class Users(db.Model, UserMixin):
    __table_args__ = {
        'autoload': True,
        'autoload_with': engine,
        'extend_existing': True
    }
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255))
    mobile = db.Column(db.BigInteger)
    email = db.Column(db.String(255))
    password = db.Column(db.String(255))
    birthday = db.Column(db.Date)
    gender = db.Column(db.String(255))
    weight = db.Column(db.String(255))
    height = db.Column(db.String(255))

    upload = db.relationship('Uploads', backref='uploader')

    def __init__(self, name, mobile, email, password, birthday, gender, weight, height):
        self.name = name
        self.email = email
        self.mobile = mobile
        self.password = password
        self.birthday = birthday
        self.weight = weight
        self.height = height
        self.gender = gender


# class Prescriptions(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     dr_name = db.Column(db.String(255))
#     dr_num = db.Column(db.String(255))
#     prescription_date = db.Column(db.Date)
#     drugs = db.Column(db.String(255))
#     upload_id = db.Column(db.Integer, db.ForeignKey("uploads.id"), nullable=False)
#     # prescriptions = db.relationship('Prescriptions', backref='user', lazy=True)
#     # prescriptions = db.relationship('Prescriptions')
#
#     def __init__(self, dr_name, dr_num, prescription_date, drugs, upload_id):
#         self.dr_name = dr_name
#         self.dr_num = dr_num
#         self.prescription_date = prescription_date
#         self.drugs = drugs
#         self.upload_id = upload_id
class Prescriptions(db.Model):
    __tablename__ = "prescriptions"
    __table_args__ = {
        'autoload': True,
        'autoload_with': engine,
        'extend_existing': True
    }
    id = db.Column(db.Integer, primary_key=True)
    dr_name = db.Column(db.String(255))
    dr_num = db.Column(db.String(255))
    prescription_date = db.Column(db.Date)
    drugs = db.Column(db.String(255))
    address = db.Column(db.String(255))
    zipcode = db.Column(db.String(255))
    upload_id = db.Column(db.Integer, db.ForeignKey("uploads.id"), nullable=False)
    # uploads = db.relationship('uploads', backref='uploads')

    def __init__(self, dr_name, dr_num, prescription_date, drugs, address, zipcode, upload_id):
        self.dr_name = dr_name
        self.dr_num = dr_num
        self.prescription_date = prescription_date
        self.drugs = drugs
        self.address = address
        self.zipcode = zipcode
        self.upload_id = upload_id

    def create(self):
        db.session.add(self)
        db.session.commit()

    def update(self):
        db.session.commit()


def run_update():
    while True:

        models = Prescriptions.query.filter_by(active=True).all()
        for model in models:
            do_something(model)
            time.sleep(300)
        db.session.close()


def do_something():
    longtask = threading.Thread(target=long_task)
    longtask.start()


def long_task():
    # // Update database here again
    # // Task has finished before it is run again.
    time.sleep(10)


class Bmis(db.Model):
    __table_args__ = {
        'autoload': True,
        'autoload_with' : engine,
        'extend_existing': True
    }
    id = db.Column(db.Integer, primary_key=True)
    weight = db.Column(db.Integer)
    height = db.Column(db.Integer)
    bmi = db.Column(db.Float)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)

    def __init__(self, weight, height, bmi, user_id):
        self.weight = weight
        self.height = height
        self.bmi = bmi
        self.user_id = user_id


class Uploads(db.Model):
    __table_args__ = {
        'autoload': True,
        'autoload_with' : engine,
        'extend_existing': True
    }
    id = db.Column(db.Integer, primary_key=True)
    image_name = db.Column(db.String(255))
    mimetype = db.Column(db.String(255))
    diagnosis = db.Column(db.String(255))
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    prescription = relationship('Prescriptions', backref=db.backref('prescription' , lazy = 'joined'), lazy='dynamic')

    def __init__(self, image_name, mimetype, diagnosis, user_id):
        self.image_name = image_name
        self.mimetype = mimetype
        self.diagnosis = diagnosis
        self.user_id = user_id

    def create(self):
        db.session.add(self)
        db.session.commit()

    def update(self):
        db.session.commit()

# Prescriptions = Table('prescriptions', metadata,
#     Column('id',Integer(), primary_key=True),
#     Column('dr_name',String(255)),
#     Column('dr_num',String(255)),
#     Column('prescription_date',DateTime),
#     Column('drugs',String(255)),
#     Column('user_id',ForeignKey('users.id')),
#     Column()
# )
#
#
#
#
#
# prescriptions = db.relationship('Prescriptions', backref='user', lazy=True)

# users = Table('users', metadata,
#               Column('id', Integer(), primary_key=True),
#               Column('name', String(255), nullable=False),
#               Column('mobile', BigInteger, nullable=False),
#               Column('email', String(255), nullable=False),
#               Column('Password', String(255), nullable=False),
#               Column('birthday', DateTime, nullable=True),
#               Column('weight', String(255), nullable=True),
#               Column('height', String(255), nullable=True),
#               )
