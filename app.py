from flask import Flask, request, jsonify, render_template
from static.models.predict import App

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    text = [request.form['tweet_text']][0]
    predict = App(text)
    probability = predict.calculate()
    return render_template('index.html', text = text, prediction='Probability of tweet being about real disaster is {}'.format(probability))

if __name__ == "__main__":
    app.run(debug=True)