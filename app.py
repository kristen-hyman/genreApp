import os
import numpy as np
import flask
import pickle
from flask import Flask, render_template, request
from model import lyric_predict
import model

#creating instance of the class
app=Flask(__name__)

#run the model while starting flask
model.model_start()

#model = pickle.load(open('model.pkl','rb'))
def ValuePredictor(lyrics):
    #loaded_model = pickle.load(open("model.pkl","rb"))
    result = model.lyric_predict(lyrics)
    
    return result

#to tell flask what url shoud trigger the function index()
@app.route('/')
def index():
    return flask.render_template('index.html')

@app.route('/aboutUs.html')
def about():
    return flask.render_template('aboutUs.html')

@app.route('/naiveBayesinfo.html')
def info():
    return flask.render_template('naiveBayesinfo.html')


@app.route('/result.html',methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        result = ValuePredictor(to_predict_list)
        return render_template("result.html", prediction=result)

    return 
    
if __name__ == "__main__":
    app.run(debug=True)

