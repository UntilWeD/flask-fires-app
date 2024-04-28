import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from flask import Flask, render_template
import googleapiclient.discovery

#flask --app fires_flask run

app  = Flask (__name__)
app.config['SECRET_KEY'] = 'hard to guess string'

from flask_bootstrap import Bootstrap5
bootstrap = Bootstrap5(app)

from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
from sklearn.model_selection import train_test_split
STRING_FIELD = StringField('max_wind_speed', validators=[DataRequired()])

class LabForm (FlaskForm):
    longitude = StringField('longitude (1-7)', validators=[DataRequired()]) 
    latitude = StringField('latitude (1-7)', validators=[DataRequired()]) 
    month = StringField('month (01-Jan~ Dec-12)', validators=[DataRequired()]) 
    day = StringField('day(00-sun ~ 06-sat, 07-hol)', validators=[DataRequired()])
    avg_temp = StringField('avg_temp', validators=[DataRequired()]) 
    max_temp = StringField('max_temp', validators = [DataRequired()])
    max_wind_speed = StringField('max_wind_speed', validators=[DataRequired()]) 
    avg_wind = StringField('avg_wind', validators=[DataRequired()]) 
    submit = SubmitField('Submit')

@app.route('/')
@app.route('/index') 
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def lab():
    form = LabForm()
    if form.validate_on_submit():
        X_test = [[float (form.longitude.data),
                float(form.latitude.data),
                str(form.month.data),
                str(form.day.data),
                float(form.avg_temp.data),
                float(form.max_temp.data),
                float(form.max_wind_speed.data),
                float(form.avg_wind.data)]]


        # in order to make a prediction,
        # we must scale the data using the same scale as the one used to make model 
        # get the data for the fires data.
        X_test = pd.DataFrame(X_test, columns=['longitude', 'latitude', 'month', 'day',
                            'avg_temp', 'max_temp', 'max_wind_speed', 'avg_wind'])

                    
        fires= pd.read_csv("./sanbul2district-divby100.csv", sep=",")
        train_set, test_set = train_test_split(fires, test_size=0.2, random_state=42) 
        fires_train = train_set.drop(['burned_area'], axis=1)
        fires_train_num = fires_train.drop(['month', 'day'], axis=1)

        num_pipeline = Pipeline([
            ('std_scaler', StandardScaler()),
        ])

        num_attribs = list(fires_train_num) 
        cat_attribs = ['month', 'day']

        full_pipeline = ColumnTransformer ([
            ('num', num_pipeline, num_attribs),
            ('cat', OneHotEncoder(), cat_attribs),
        ])

        full_pipeline.fit(fires_train)
        X_test = full_pipeline.transform(X_test)

        # create the resource to the model web api on GCP
        model_id = "my_fires_model"
        project_id = 'myfirstproject-399223'
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "myfirstproject-399223-dc2cfc445a59.json"



        model_path = "projects/{}/models/{}".format(project_id, model_id) 
        model_path += "/versions/v0001/" # if you want to run a specific version 
        ml_resource = googleapiclient.discovery.build("ml", "v1").projects()

        # format the data as a json to send to the web api 
        input_data_json = {"signature_name": "serving_default",
                            "instances": X_test.tolist()}

        # make the prediction
        request = ml_resource.predict(name=model_path, body=input_data_json)
        response = request.execute()
        print("\nresponse: \n", response)

        if "error" in response:
            raise RuntimeError (response ["error"])
        
        # extract the prediction from the response
        predD = np.array([pred['dense_1'] for pred in response["predictions"]]) 
        #A/model.summary() -> 마지막 레이어 참조!!

        print(predD[0][0])
        res = predD[0][0]
        np.round(res, 2)
        res = (float) (np.round(res*100))
        res = np.round(res, 2)

        return render_template('result.html', res=res)
    
    return render_template('prediction.html', form=form)
    
if __name__== '__main__':
    app.run()
