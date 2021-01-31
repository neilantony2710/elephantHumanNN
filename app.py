from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import sklearn.model_selection
import keras
from keras.models import load_model

app = Flask(__name__)
model = load_model("humanelephantrecog.h5")


def prediction(w, h):
    h = int(h)
    w = int(w)
    inp = [w, h]

    inp = np.array(inp)
    inp = np.reshape(inp, (1, 2))
    pred = model.predict(inp)
    pred = np.argmax(pred)
    if pred == 0:
        return("Human!")
    else:
        return("Elephant!")


@app.route('/', methods=['GET', 'POST'])
def index():

    if request.method == "GET":

        return render_template("index.html")
    else:

        weight = request.form["weight"]
        height = request.form["height"]

        pr = prediction(weight, height)
        return render_template("prediction.html", display=pr)


if __name__ == '__main__':
    app.run()
