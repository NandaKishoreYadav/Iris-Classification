import numpy as np
from flask import *
import pickle
app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/predict',methods=["GET","POST"])
def predict():
    sl = float(request.form.get('sl'))
    sw = float(request.form.get('sw'))
    pl = float(request.form.get('pl'))
    pw = float(request.form.get('pw'))
    pred=np.array([[sl,sw,pl,pw]])
    k=model.predict(pred)
    return render_template('pred.html',ans=k)

if __name__=='__main__':
    app.run(debug=True)