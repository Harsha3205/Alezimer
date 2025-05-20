from flask import Flask, render_template, request, redirect, url_for,session
import warnings
warnings.filterwarnings("ignore")
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd

app = Flask(__name__)
app.secret_key = 'alzheimer_disease'

dic = {0: 'MildDemented', 1: 'NonDemented', 2: 'VeryMildDemented'}

model = load_model(filepath="models/Convolutional_Neural_Network.h5")

filepath="ValidationImages/mild_demented/milddemented (1).jpg"


def predict_label(img_path):
    image = load_img(img_path, grayscale=True, color_mode='grayscale', target_size=(128, 128))
    image = img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    model_pred = model.predict(image)
    proba = model_pred[0]
    res = np.argmax(model_pred[0])
    res = dic[res]
    return res, proba

# routes


@app.route('/')
@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        email = request.form["email"]
        pwd = request.form["password"]
        r1 = pd.read_excel('user.xlsx')
        for index, row in r1.iterrows():
            if row["email"] == str(email) and row["password"] == str(pwd):
                return redirect(url_for('home'))
        else:
            mesg = 'Invalid Login Try Again'
            return render_template('login.html', msg=mesg)
    return render_template('login.html')


@app.route('/register', methods=['POST', 'GET'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['Email']
        password = request.form['Password']
        col_list = ["name", "email", "password"]
        r1 = pd.read_excel('user.xlsx', usecols=col_list)
        new_row = {'name': name, 'email': email, 'password': password}
        r1 = r1.append(new_row, ignore_index=True)
        r1.to_excel('user.xlsx', index=False)
        print("Records created successfully")
        # msg = 'Entered Mail ID Already Existed'
        msg = 'Registration Successful !! U Can login Here !!!'
        return render_template('login.html', msg=msg)
    return render_template('register.html')


@app.route("/home", methods=['GET', 'POST'])
def home():
   return render_template("home.html")


@app.route("/submit", methods=['GET', 'POST'])
def get_hours():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/" + img.filename
        img.save(img_path)
        p = predict_label(img_path)
        prediction = p[0]
        p_score = p[1]
        p1 = []
        for i in p_score:
            a = i*100
            p1.append(float(str(a)[0:4]))
        print(p1)
        print(p_score)
        return render_template("home.html", prediction=prediction, p_score=p1, img_path=img_path)


@app.route('/password', methods=['POST', 'GET'])
def password():
    if request.method == 'POST':
        current_pass = request.form['current']
        new_pass = request.form['new']
        verify_pass = request.form['verify']
        r1 = pd.read_excel('user.xlsx')
        for index, row in r1.iterrows():
            if row["password"] == str(current_pass):
                if new_pass == verify_pass:
                    r1.replace(to_replace=current_pass, value=verify_pass, inplace=True)
                    r1.to_excel("user.xlsx", index=False)
                    msg1 = 'Password changed successfully'
                    return render_template('password_change.html', msg1=msg1)
                else:
                    msg2 = 'Re-entered password is not matched'
                    return render_template('password_change.html', msg2=msg2)
        else:
            msg3 = 'Incorrect password'
            return render_template('password_change.html', msg3=msg3)
    return render_template('password_change.html')


@app.route('/graphs', methods=['POST', 'GET'])
def graphs():
    return render_template('graphs.html')


@app.route('/cnn')
def cnn():
    return render_template('cnn.html')


@app.route('/vgg')
def vgg():
    return render_template('vgg.html')


@app.route('/logout')
def logout():
    session.clear()
    msg='You are now logged out', 'success'
    return redirect(url_for('login', msg=msg))


if __name__ == '__main__':
    app.run(debug=True)
