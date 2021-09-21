import numpy as np
import pandas as pd
from flask import Response
from flask import Flask, request, jsonify, render_template ,make_response
import pickle
from matplotlib.figure import Figure
import io
import csv
import base64
from io import StringIO
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from collections import Counter

flask_app = Flask(__name__)
model = pickle.load(open('KMeans.pkl', 'rb'))

@flask_app.route('/')
def home():
    return render_template('index.html')

@flask_app.route('/predict',methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    if(output==0):
        return render_template('index.html', prediction_text='Label ; 0' )
    elif(output==1):
        return render_template('index.html', prediction_text='Label: 1' )
    elif(output==2):
        return render_template('index.html', prediction_text='Label: 2' )
    elif(output==3):
        return render_template('index.html', prediction_text='Label: 3' )
    else:
        return render_template('index.html', prediction_text='Label: 4' )

@flask_app.route('/defaults',methods=['POST'])
def defaults():
    return render_template('index.html')
@flask_app.route('/default',methods=["POST"])
def default():

    return render_template('layout.html')

def transform(text_file_contents):
    return text_file_contents.replace("=", ",")

@flask_app.route('/graph', methods=["POST"])
def graph():
    f = request.files['data_file']
 
    if not f:
 
        return "No file"
 
 
    stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
 
    csv_input = csv.reader(stream)
 
    print(csv_input)
 
    for row in csv_input:
 
        print(row)
 
 
    stream.seek(0)
 
    result = transform(stream.read())
 
    
    if request.form['class'] == 'kmeans':
        df = pd.read_csv(StringIO(result))

        loaded_model = pickle.load(open('kMeans.pkl', 'rb'))
        kmeans = loaded_model.fit(df)
        cluster = kmeans.predict(df)
        s = silhouette_score(df, kmeans.labels_)
        center = Counter(kmeans.labels_)
        fig = plt.figure(figsize=(5, 4))
        plt.scatter(df['Annual Income (k$)'],df['Spending Score (1-100)'],c=cluster, cmap='Paired')
        plt.title('KMeans')
        imgdata = StringIO()
        fig.savefig(imgdata, format='svg')
        imgdata.seek(0)
        data = imgdata.getvalue()

    elif request.form['class'] == 'agglomerative':
        df = pd.read_csv(StringIO(result))

        loaded_model = pickle.load(open('agglo.pkl', 'rb'))
 
        cluster = loaded_model.fit_predict(df)
        s = silhouette_score(df, loaded_model.labels_)
        center = Counter(loaded_model.labels_)
        fig = plt.figure(figsize=(5, 4))
        plt.scatter(df['Annual Income (k$)'],df['Spending Score (1-100)'],c=cluster, cmap='Paired')
        plt.title('Agglomerative')
        imgdata = StringIO()
        fig.savefig(imgdata, format='svg')
        imgdata.seek(0)
        data = imgdata.getvalue()

    elif request.form['class'] == 'dbscan':
        df = pd.read_csv(StringIO(result))

        loaded_model = pickle.load(open('dbscan.pkl', 'rb'))
        labeldb=loaded_model.fit_predict(df)
        cluster = loaded_model.labels_
        s = silhouette_score(df, labeldb)
        center = Counter(loaded_model.labels_)
        fig = plt.figure(figsize=(10, 7))
        plt.scatter(df['Annual Income (k$)'],df['Spending Score (1-100)'],c=cluster, cmap='Paired')
        plt.title('DBSCAN')
        imgdata = StringIO()
        fig.savefig(imgdata, format='svg')
        imgdata.seek(0)
        data = imgdata.getvalue()
    return render_template('layout.html',image=data)
@flask_app.route('/transformdb', methods=["POST"])
def transformdb_view():
    f = request.files['data_file']
    if not f:
        return "No file"
    stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
    csv_input = csv.reader(stream)
    result = transform(stream.read())
    df = pd.read_csv(StringIO(result))
    loaded_model = pickle.load(open('dbscan.pkl', 'rb'))
    if request.form['class'] == 'dbscore':
        labeldb=loaded_model.fit_predict(df)
        return render_template('layout.html', prediction_text='silhouette_score :{} '.format(silhouette_score(df,labeldb)))
    if request.form['class'] == 'dbscan':
        df['cluster'] = loaded_model.labels_
        response = make_response(df.to_csv())
        response.headers["Content-Disposition"] = "attachment; filename=DBscan.csv"
        return response
    if request.form['class'] == 'dbgraph':
        cluster = loaded_model.labels_
        center = Counter(loaded_model.labels_)
        fig = plt.figure(figsize=(10, 7))
        plt.scatter(df['Annual Income (k$)'],df['Spending Score (1-100)'],c=cluster, cmap='Paired')
        plt.title('DBSCAN')
        imgdata = StringIO()
        fig.savefig(imgdata, format='svg')
        imgdata.seek(0)
        data = imgdata.getvalue()
        return render_template('graph.html',image=data)
@flask_app.route('/scoredb', methods=["POST"])
def scoredb_view():

    loaded_model = pickle.load(open('dbscan.pkl', 'rb'))
    labeldb=loaded_model.fit_predict(df)
    return render_template('layout.html', prediction_text='silhouette_score :{} '.format(silhouette_score(df,labeldb)))

@flask_app.route('/transformagglo', methods=["POST"])
def transformagglo():
    f = request.files['data_file']
    if not f:
        return "No file"

    stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
    csv_input = csv.reader(stream)
    result = transform(stream.read())

    df = pd.read_csv(StringIO(result))
    loaded_model = pickle.load(open('agglo.pkl', 'rb'))
    if request.form['class'] == 'aggloscore':
        labelagglo=loaded_model.fit_predict(df)
        return render_template('layout.html', prediction_text='silhouette_score :{} '.format(silhouette_score(df,labelagglo)))
    if request.form['class'] == 'agglo':
        df['cluster'] = loaded_model.fit_predict(df)
        response = make_response(df.to_csv())
        response.headers["Content-Disposition"] = "attachment; filename=Agglomerative Clustering.csv"
        return response
    if request.form['class'] == 'agglograph':
        cluster = loaded_model.fit_predict(df)
        center = Counter(loaded_model.labels_)
        fig = plt.figure(figsize=(10, 7))
        plt.scatter(df['Annual Income (k$)'],df['Spending Score (1-100)'],c=cluster, cmap='Paired')
        plt.title('Agglomerative')
        imgdata = StringIO()
        fig.savefig(imgdata, format='svg')
        imgdata.seek(0)
        data = imgdata.getvalue()
        return render_template('graph.html',image=data)

@flask_app.route('/transformKM', methods=["POST"])
def transformKM_view():
    f = request.files['data_file']
    if not f:
        return "No file"
    stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
    csv_input = csv.reader(stream)
    result = transform(stream.read())
    df = pd.read_csv(StringIO(result))
    loaded_model = pickle.load(open('KMeans.pkl', 'rb'))
    if request.form['class'] == 'kmeanscore':
        labelkmeans=loaded_model.fit_predict(df)
        return render_template('layout.html', prediction_text='silhouette_score :{} '.format(silhouette_score(df,labelkmeans)))
    if request.form['class'] == 'kmeans':
        df['cluster'] = loaded_model.fit_predict(df)
        response = make_response(df.to_csv())
        response.headers["Content-Disposition"] = "attachment; filename=KMeans.csv"
        return response
    if request.form['class'] == 'kmeansgraph':
        kmeans = loaded_model.fit(df)
        cluster = kmeans.predict(df)
        center = Counter(kmeans.labels_)
        fig = plt.figure(figsize=(10, 7))
        plt.scatter(df['Annual Income (k$)'],df['Spending Score (1-100)'],c=cluster, cmap='Paired')
        plt.title('KMeans')
        imgdata = StringIO()
        fig.savefig(imgdata, format='svg')
        imgdata.seek(0)
        data = imgdata.getvalue()
        return render_template('graph.html',image=data)

if __name__ == "__main__":
    flask_app.run(debug=True)


