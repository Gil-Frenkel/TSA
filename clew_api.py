from flask import Flask, request, redirect, url_for, flash, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import joblib

app = Flask(__name__)

# this function posts the raw data to the server and response a prediction result
@app.route('/api', methods=['POST'])
def getPatientLabel():
	try:
		dataFile = request.args['input']
	except KeyError:
		return "Error! Data input is not provided."
	
	finally:
		# load the feature scaler
		scaler = joblib.load('/home/gil/clew/scaler.dat')
		# fetch the maximum sequence size for zero padding
		max_seq_size = 60
		# load the prediction model
		model = load_model('/home/gil/clew/model.h5')

		# load the data
		data = pd.read_csv(dataFile)
		data = data[['patient_id','measurement_x','measurement_y','measurement_z']]
		# process the data for the model
		measurements_label = ['measurement_x','measurement_y','measurement_z']
		## feature Min-Max rescaling
		data[measurements_label] = scaler.transform(data[measurements_label])
		## Fill missing values with -1
		data.fillna(-1, inplace=True)
		## perform zero padding using max_seq_size
		lst = []
		for pat in data.patient_id.unique():
			pat_data = data[data.patient_id == pat][measurements_label].copy()
			pat_data = np.asarray(pat_data).reshape((1,) + pat_data.shape)
			n = pat_data.shape[1]
			if n < max_seq_size:
				completion = np.zeros(shape=(1,max_seq_size-n,3))
				pat_data = np.concatenate([completion, pat_data], axis=1)
			else:
				pat_data = pat_data[:,-max_seq_size:,:]
			lst.append(pat_data)
		X = np.concatenate(lst, axis=0)

		# predict new labels from processed data
		labels = model.predict(X).reshape(-1).astype('int64').tolist()
		patients = data.patient_id.unique().tolist()
		prediction = {"predictions": {p:l for p,l in zip(patients, labels)}}

		# return the flask
		res = jsonify(prediction)

		return res

if __name__ == '__main__':
	app.run(debug=True, host='127.0.0.1', port=5000)