# from flask_mysqldb import MySQL
import matplotlib
from flask import Flask, render_template, request, send_file, redirect, url_for, session, flash, jsonify
from json import dump
# import flask_mysqldb
from pickle import GET
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func
import pytesseract
import numpy as np
import pandas as pd
import re
import cv2
import geocoder
from matplotlib import pyplot as plt
from pprint import pprint
from dateparser.search import search_dates
from dateparser import parser
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
import sqlalchemy as db
from sqlalchemy import exists
# from sqlalchemy.testing.suite.test_reflection import users
from sqlalchemy.orm import sessionmaker
from models import Prescriptions, Users, Uploads, Bmis
from werkzeug.utils import secure_filename
import os
import json
from settings import app, db, engine, drugname
import spacy
import chat
import requests
from isodate import parse_duration
from paddleocr import PaddleOCR
from autocorrect import Speller
from flask_socketio import SocketIO

socketio = SocketIO(app)

connection = engine.connect()
metadata = db.MetaData()

Session = sessionmaker(bind=engine)
session = Session()
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
spellcheck = Speller(only_replacements=True)
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/Cellar/tesseract/5.1.0/bin/tesseract"


@login_manager.user_loader
def load_user(user_id):
    return Users.query.get(int(user_id))


@app.route('/registration', methods=['GET', 'POST'])
def registration():
    if request.method == 'POST':
        email = request.form.get('email')
        # census = db.Table('users', metadata, autoload=True, autoload_with=engine)
        user = db.session.query(db.exists().where(Users.email == email)).scalar()
        # user = db.select([census]).where(census.columns.email == email)
        # found_user = Users.query.filter_by(email = email).first()
        if user:
            flash("Email already exists!!!!", 'danger')
            return redirect(url_for('login'))
        else:
            email = request.form.get('email')
            name = request.form.get('name')
            password = request.form.get('password')
            birthday = request.form.get('dob')
            gender = request.form['gender']
            weight = request.form.get('weight')
            height = request.form.get('height')
            mobile = request.form.get('mobile')
            usr = Users(name, mobile, email, password, birthday, gender, weight, height)
            db.session.add(usr)
            db.session.commit()
            # session['id'] = id

            flash('Registered Successfully!', 'info')
            return redirect(url_for('home'))
    return render_template('registration.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    username = request.form.get('email')
    pass1 = request.form.get('password')
    user = Users.query.filter_by(email=username).first()
    if username is not None:
        if user:
            found_user = Users.query.filter_by(email=username, password=pass1).first()
            if found_user:
                login_user(user)
                session.permanent = False

                return redirect(url_for('home'))
            else:
                flash("Wrong Password", "danger")
                return redirect(url_for('login'))
        else:
            flash("User doesn't exist!!!")
            return render_template("login.html")
    return render_template("login.html")


@app.route('/<dr_name>', methods=['GET', 'POST'])
def dr_name(dr_name):
    gender = current_user.gender
    if gender == "male":
        avatar = 'images/male.png'
    else:
        avatar = 'images/female.png'
    # userData = Users.query.filter_by(id=id).first()
    prescription = db.Table('prescriptions', metadata, autoload=True, autoload_with=engine)
    upload = db.Table('uploads', metadata, autoload=True, autoload_with=engine)
    dcount1 = db.session.query(
        upload, prescription).filter(prescription.columns.dr_name == dr_name).filter(prescription.columns.upload_id == upload.columns.id).filter(upload.columns.user_id == current_user.id).all()
    # file_name = db.select(uploads.columns.image_name).where(uploads.columns.user_id == current_user.id)
    # filename = db.select(uploads.columns.upload_id from prescriptions where (select prescriptions.columns.user_id FROM uploads WHERE prescriptions.columns.user_id = current_user.id))
    # resultproxy = connection.execute(file_name)
    # resultset = resultproxy.fetchall()

    # date = datetime.fromtimestamp(Prescriptions.prescription_date)
    return render_template('prescription.html', userData = current_user, drName = dcount1, avatar = avatar, dr_name= dr_name)


@app.route('/logout', methods=['GET', 'POST'])
@login_required
@socketio.on('disconnect')
def logout():
    logout_user()
    flash("Nikal Laude")
    # Session.pop('health1234', None)
    return redirect(url_for('login'))


nlp2 = spacy.load("/Users/viren/Documents/lakehead/4th sem/project/static/ner")


def extract_drug_entity(text):
    docx = nlp2(text)
    result = [(ent, ent.label_) for ent in docx.ents]
    return result


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    image = request.files['image']
    diagnosis = request.form.get('diagnosis')
    if not image:
        return 'no image uploaded', 400
    filename = secure_filename(image.filename)
    mimetype = image.mimetype

    path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
    image.save(path)
    img = Uploads(image_name=filename, mimetype=mimetype,
                  diagnosis=diagnosis, user_id=current_user.id)

    db.session.add(img)
    db.session.commit()
    img_1 = cv2.imread(path)
    gray_image = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    threshold_img = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))

# Applying dilation on the threshold image
    dilation = cv2.dilate(threshold_img, rect_kernel, iterations = 1)

    # Finding contours
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                                     cv2.CHAIN_APPROX_NONE)

    # Creating a copy of image
    im2 = img_1.copy()

    # A text file is created and flushed
    file = open("recognized.txt", "w+")
    file.write("")
    file.close()

    # Looping through the identified contours
    # Then rectangular part is cropped and passed on
    # to pytesseract for extracting text from it
    # Extracted text is then written into the text file
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Drawing a rectangle on copied image
        rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Cropping the text block for giving input to OCR
        cropped = im2[y:y + h, x:x + w]

        # Apply OCR on the cropped image
        extractedInformation = pytesseract.image_to_string(cropped)

    # extractedInformation = pytesseract.image_to_string(threshold_img)
    # print(result)
    with open("trial1.txt", 'w') as f:
        f.write(extractedInformation)
    final_list = extractedInformation.split('\n')
    prediction = map(extract_drug_entity, final_list)
    zip_pattern = re.compile('[A-Za-z]\d[A-Za-z][ ]?\d[A-Za-z]\d')
    pincode = zip_pattern.search(extractedInformation).group()
    ad_pattern = re.compile('\d+(?:[A-Za-z0-9.-]+[ ]?)+(?:Avenue|Lane|Road|Boulevard|Drive|Street|Ave|Dr|Rd|Blvd|Ln|St)\.?')
    address = ad_pattern.search(extractedInformation).group()
    final_address = address +'\n' + pincode
    # prediction = extract_drug_entity(extractedInformation)
    numbersSquare = list(prediction)
    list2 = [x for x in numbersSquare if x]
    dr = [tup[0] for tup in list2]
    dr1 = [tup[0] for tup in dr]
    dr2 = [str(x).lower() for x in dr1]
    print(dr2)
    extractedInformation = extractedInformation.strip()
    extractedInformation = re.sub('\\s+Or. | \\s+Dr. ', ' Dr. ', extractedInformation)
    dr_name = re.split('Dr. ', extractedInformation)[1].split('\n')[0]
    dr_num_pattern = re.compile('(?:\+?(\d{1,3}))?[-. (]*(\d{3})[-. )]*(\d{3})[-. ]*(\d{4})(?: *x(\d+))?')
    dr_num = dr_num_pattern.search(extractedInformation)
    dr_num = dr_num.group(0)
    dates = search_dates(extractedInformation, settings={'STRICT_PARSING': True})
    dates_df = pd.DataFrame(dates)
    filter = dates_df[0].str.contains('\d{5}')
    filter_dates = dates_df[~filter]
    sorted_dates = list(filter_dates[1].sort_values())
    # birth_date = sorted_dates[0]
    prescription_date = sorted_dates[1]
    drg = list(set(dr2).intersection(drugname))
    drugs = np.array(drg)
    pres_data = Prescriptions(dr_name=dr_name, dr_num=dr_num, prescription_date=prescription_date, drugs=drugs, address = final_address, zipcode = pincode, upload_id=img.id)
    # with open('trial.txt','w') as f:
    #     f.write(str(pres_data.upload_id))

    db.session.add(pres_data)
    db.session.commit()

    return redirect(url_for('home'))


@app.route('/', methods=['GET'])
def splash():
    return render_template("splash.html")


@app.route('/bmi1', methods=['GET', 'POST'])
def bmi1():
    h = int(current_user.height)
    w = int(current_user.weight)
    bmi_info = round((w/(h**2))*10000,2)
    return render_template("bmi2.html", bmi=bmi_info)


@app.route('/bmi', methods=['GET', 'POST'])
def bmi():
    matplotlib.pyplot.switch_backend('Agg')
    gender = current_user.gender
    if gender == "male":
        avatar = 'images/male.png'
    else:
        avatar = 'images/female.png'
    engine = db.create_engine('mysql://root@127.0.0.1/trial', {})
    connection = engine.connect()
    bmi_table = db.Table('bmis', metadata, autoload= True, autoload_with=engine)
    query1 = db.select([bmi_table.columns.bmi, bmi_table.columns.created_at]).where(bmi_table.columns.user_id == current_user.id)
    graph = connection.execute(query1).fetchall()
    x=[]
    y=[]
    for i in graph:
        x.append(i[1])
        y.append(i[0])
    plt.plot(x, y)
    plt.xlabel("Period")
    plt.ylabel("Bmi Ratio")
    plt.savefig('/Users/viren/Documents/lakehead/4th sem/project/static/graph/bmi_graph.png')
    graph1 = 'graph/bmi_graph.png'
    if request.method == 'POST':
        weight = float(request.form.get('weight'))
        height = float(request.form.get('height'))
        bmi = round((weight / ((height / 100) ** 2)), 2)
        bmi_table = Bmis(weight=weight, height=height, bmi=bmi, user_id=current_user.id)
        db.session.add(bmi_table)
        db.session.commit()
        census = db.Table('users', metadata, autoload=True, autoload_with=engine)
        updated = db.update(census).values(height=height, weight=weight)
        query = updated.where(census.columns.id == current_user.id)
        results = connection.execute(query)

    return render_template('bmi.html', avatar = avatar, url=graph1)


@app.route('/profile/<int:id>', methods=['GET', 'POST'])
def profile(id):
    engine = db.create_engine('mysql://root@127.0.0.1/trial', {})
    connection = engine.connect()
    census = db.Table('users', metadata, autoload=True, autoload_with=engine)
    update_user = Users.query.get_or_404(id)
    pas = current_user.password
    print(pas)
    gender = current_user.gender
    if gender == "male":
        avatar = 'images/male.png'
    else:
        avatar = 'images/female.png'

    if request.method == 'POST':
        update_user.name = request.form.get('name')
        # print(update_user.name)
        update_user.email = request.form.get('email')
        update_user.password = request.form.get('new_password')
        if update_user.password == '':
            update_user.password = pas
        update_user.birthday = request.form.get('birthday')
        update_user.gender = request.form.get('gender')
        update_user.weight = request.form.get('weight')
        update_user.height = request.form.get('height')
        update_user.mobile = request.form.get('mobile')

        updated = db.update(census).values(name=update_user.name, email=update_user.email,
                                           password=update_user.password, mobile=update_user.mobile,
                                           birthday=update_user.birthday, gender=update_user.gender,
                                           height=update_user.height, weight=update_user.weight)
        query = updated.where(census.columns.id == id)
        results = connection.execute(query)

        # up_user = Users.query.filter_by(id).update(update_user.name,  update_user.mobile, update_user.email, update_user.password,update_user.birthday, gender, update_user.weight, update_user.height)
        db.session.commit()

    return render_template("profile_page.html", avatar=avatar)


@app.route('/home', methods=['GET', 'POST'])
@login_required
def home():
    engine = db.create_engine('mysql://root@127.0.0.1/trial', {})
    connection = engine.connect()
    u_id = current_user.id

    u_name = current_user.name
    # query = db.session.query(
    #     db.func.count(Prescriptions.dr_name),
    #     Prescriptions.dr_name, Prescriptions.dr_num,
    # ).group_by(
    #     Prescriptions.dr_name
    # ).having(
    #     db.func.count(Prescriptions.dr_name) >= 1
    # ).all()

    # result = db.session.query(Prescriptions.dr_name).join(Prescriptions, Prescriptions.upload_id == Uploads.id).filter(Uploads.user_id == current_user.id).all()
    # census = db.Table('uploads', metadata, autoload=True, autoload_with=engine)
    # userdata = db.select([census]).where(census.columns.user_id == current_user.id)
    # data = Uploads.query.filter(Uploads.user_id == u_id).all()
    # data = db.session.query(Uploads.prescription.dr_name).all()
    # resultproxy = connection.execute("Select * from prescriptions WHERE upload_id IN (SELECT id from uploads WHERE user_id = ' + u_id +')")
    # resultset = resultproxy.fetchall()
    # data = db.session.query(Users, Uploads).join(Users, Users.id == Uploads.user_id).all()
    # upload_data = data.query.filter_by(Users.id == u_id).all()
    # cnt = resultset.count()
    # data = db.session.query(u).all()

    census = db.Table('prescriptions', metadata, autoload=True, autoload_with=engine)
    upload = db.Table('uploads', metadata, autoload=True, autoload_with=engine)
    # data = db.select([census.columns.dr_name]).where(census.columns.upload_id.in_(db.select([uploads.columns.id]).where(uploads.columns.user_id == u_id))).group_by(census.columns.dr_name)
    dcount = db.session.query(db.func.count(census.columns.dr_name), census.columns.dr_name,
                              census.columns.dr_num).where(
        census.columns.upload_id.in_(db.select([upload.columns.id]).where(upload.columns.user_id == u_id))).group_by(
        census.columns.dr_name).having(db.func.count(census.columns.dr_name) >= 1).all()
    # data1 = connection.execute(db.select([census]).where(census.columns.upload_id.in_(db.select([uploads.columns.id]).where(uploads.columns.user_id == u_id))).group_by(census.columns.dr_name).having(db.func.count(census.columns.dr_name)>=1))
    # db.select([db.func.sum(census.columns.pop2008).label('pop2008'), census.columns.sex]).group_by(census.columns.sex)
    gender = current_user.gender
    # result2 = data1.fetchall()
    # dcnt = db.select([census, uploads]).where(census.columns.upload_id.in_(db.select([uploads.columns.id]).where(uploads.columns.user_id == u_id)))
    # dcount1 = db.session.query(
    #     uploads, census).filter(census.columns.dr_name == dr_name).filter(census.columns.upload_id == uploads.columns.id).filter(uploads.columns.user_id == u_id).all()

    # with open('trial.txt','w') as f:
    #     f.write(str(pres_data.upload_id))
    # results = dcnt.prescription.image_name
    # result1 = connection.execute(dcount1).fetchall()
    # resultset = dcnt.fetchall()
    # results = str(results)
    if gender == "male":
        avatar = 'images/male.png'
    else:
        avatar = 'images/female.png'

    # with open('trial.txt', 'w') as f:
    #     f.write(str(dcount1))
    # result = connection.execute(data)
    # resultset = result.fetchall()
    # db.select([census.columns.state, census.columns.sex]).where(census.columns.state.in_(['Texas', 'New York']))
    # query = db.select([db.func.count(census.columns.dr_name), census.columns.dr_name, census.columns.dr_num]).group_by(census.columns.dr_name).having(db.func.count(census.columns.dr_name) >= 1)
    # doctor = db.session.query(Prescriptions.dr_name).distinct().all()

    # query = db.select([db.func.sum(census.columns.dr_name)]).group_by(census.columns.dr_name)
    # # query = db.select([census.columns.dr_name.count()])

    return render_template("home.html", query1=dcount, name=u_name, avatar=avatar)


@app.route('/base')
def base():
    return render_template("chat1.html")


@app.route('/chat', methods=['GET', 'POST'])
def chatbotResponse():

    if request.method == 'POST':
        the_question = request.get_json().get("message")
        response = chat.chatbot_response(the_question)
        message = {"answer" : response}

    return jsonify(message)


@app.route('/maps', methods=['GET','POST'])
def maps():
    gender = current_user.gender
    if gender == "male":
        avatar = 'images/male.png'
    else:
        avatar = 'images/female.png'
    g = geocoder.ip('me')
    loc = g.latlng
    url = "https://google-maps28.p.rapidapi.com/maps/api/place/nearbysearch/json"

    querystring = {"location": ','.join([str(n) for n in loc]) ,
                   "radius":"5000",
                   "language":"en",
                   "type":["doctor","hospital"]
                   }

    headers = {
        "X-RapidAPI-Key": "e182587402msh8fb9cb7826edf43p1eda64jsnb099563a5477",
        "X-RapidAPI-Host": "google-maps28.p.rapidapi.com"
    }

    response = requests.request("GET", url, headers=headers, params=querystring)
    data = response.json()
    return render_template('maps.html', data = data, loc=loc, avatar = avatar)


@app.route('/videos', methods=['GET', 'POST'])
def videos():
        h = int(current_user.height)
        w = int(current_user.weight)
        bmi_data = round((w/(h**2))*10000,2)
        if(bmi_data < 18.5 ):
            data = "underweight"

        elif(bmi_data > 18.5 and bmi_data < 24.9):
            data = "healthy"
        elif(bmi_data > 25 and bmi_data < 29.9):
            data = "overweight"
        else:
            data = "obese"

        search_url = 'https://www.googleapis.com/youtube/v3/search'
        videos_url = 'https://www.googleapis.com/youtube/v3/videos'
        params = {
            'key': 'AIzaSyChdx2e082EBpKVzC_NudAI9z6tpakgjSA',
            'q': 'workout for'+data+'people',
            'part': 'snippet',
            'maxResults': 6,
            'type': 'video'
        }

        r = requests.get(search_url,params=params)
        results = r.json()['items']
        video_ids = []
        videos = []
        for result in results:
            video_ids.append(result['id']['videoId'])

        video_params = {
            'key': 'AIzaSyChdx2e082EBpKVzC_NudAI9z6tpakgjSA',
            'id': ','.join(video_ids),
            'part':'snippet,contentDetails',
            'maxResults': 6

        }
        r = requests.get(videos_url,params=video_params)
        results = r.json()['items']
        for result in results:
          video_data = {
            'id' : result['id'],
            'url' : f'https://www.youtube.com/watch?v={ result["id"] }',
            'thumbnail' : result['snippet']['thumbnails']['high']['url'],
            'duration' : int(parse_duration(result['contentDetails']['duration']).total_seconds() // 60),
            'title' : result['snippet']['title'],
          }
          videos.append(video_data)
        return render_template('videos.html',videos=videos)


@app.route('/slider', methods=['GET','POST'])
def slider():
    return render_template("cardslider.html")


if __name__ == '__main__':
    app.run(debug=True)
