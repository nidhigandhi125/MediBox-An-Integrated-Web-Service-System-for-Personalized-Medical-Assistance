import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
import requests
from bs4 import BeautifulSoup
from tensorflow import keras
from tensorflow.keras.models import load_model
import json
import random
from settings import app, db, engine, drugname

connection = engine.connect()
metadata = db.MetaData()

model = load_model('chatbot_model.h5')
intents = json.loads(open('intents.json', encoding='utf-8').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

drug = db.Table('drug_data3436', metadata, autoload=True, autoload_with=engine)
drugdata = connection.execute(db.select(drug.columns.drugName))
drugname = drugdata.all()
drugname = [str(x[0]).lower() for x in drugname]

with open('td1.txt', 'w') as f:
    f.write(str(drugname))


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return (np.array(bag))


def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
        # print(return_list)
    return return_list


def getResponse(ints, intents_json, sentence):
    try:

        q1 = sentence.split()
        string = list(set(drugname).intersection(q1))
        if string:
            string = string[0]
            # http = urllib3.PoolManager()
            URL = "https://www.nhs.uk/medicines/" + string
            response = requests.get(URL)
            # response = http.request('GET', url)
            # soup = BeautifulSoup(response.data, 'html.parser')
            text = response.text
            soup = BeautifulSoup(text, 'html.parser')
            # results = soup.find(id="dosage")
            side_effects_lst = ["side-effects", "side", "effects", "effect"]
            info_list = ["dose", "dosage", "quantity", "qty", "when"]
            what_list = ["what", "about"]
            if any(item in side_effects_lst for item in q1):
                results = soup.find(id="side-effects")
            elif any(item in info_list for item in q1):
                results = soup.find(id='how-and-when-to-take-'+ string)
            elif any(item in what_list for item in q1):
                results = soup.find(id="about-" + string)

            result = [results('p')[0].text, URL]
            return result
        else:
            tag = ints[0]['intent']
            list_of_intents = intents_json['intents']
            for i in list_of_intents:
                if (i['tag'] == tag):
                    result = random.choice(i['responses'])
                    break
                else:
                    result = "You must ask the right questions"
            return result
    # results = soup.find(id ="key-facts")
    # tags = soup('p')
    # for tag in tags :
    # results = soup.find(id="dosage").contents[0]
    # results = soup.find(id ="about-"+med)
    # result1 = tag.contents[0]
    except:
        result = 'The question is out of my knowledge'
        return result


def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents, msg)
    return res
