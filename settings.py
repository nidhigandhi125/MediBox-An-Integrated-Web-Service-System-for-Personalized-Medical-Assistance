from flask import Flask
from flask_sqlalchemy import SQLAlchemy


UPLOAD_FOLDER = '/Users/viren/Documents/lakehead/4th sem/project/static/uploads'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
# TEMPLATES_AUTO_RELOAD = True
app = Flask(__name__, template_folder='template')
app.secret_key = "health1234"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root@127.0.0.1/trial'
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = True


db = SQLAlchemy(app)
engine = db.create_engine('mysql://root@127.0.0.1/trial', {})
connection = engine.connect()
metadata = db.MetaData()

drug = db.Table('drug_data3436', metadata, autoload=True, autoload_with=engine)
drugdata = connection.execute(db.select(drug.columns.drugName))
condition = connection.execute(db.select(drug.columns.condition))
condition = condition.all()
drugname = drugdata.all()
drugname = [str(x[0]).lower() for x in drugname]
condition = [str(y[0]).lower() for y in condition]
