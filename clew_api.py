from flask import Flask, request, redirect, url_for, flash, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import joblib
import json

app = Flask(__name__)

def remove_outlier_IQR(df):
	Q1=df.quantile(0.25)
	Q3=df.quantile(0.75)
	IQR=Q3-Q1
	df_final=df[(df < (Q1-1.5*IQR)) | (df > (Q3+1.5*IQR))]
	return Q1, Q3, df_final

# this function posts the raw data to the server and response a prediction result
@app.route('/api', methods=['POST'])
def getPatientLabel():
	try:
		dataFile = request.args['input']
	except KeyError:
		return "Error! Data input is not provided."
	
	finally:
		# load the learnt quantiles and maximum sequence length for zero padding
		with open('/home/gil/clew/process_params.json', 'r') as jsonFile:
			params = json.load(jsonFile)
			jsonFile.close()
		
		q1_x = params['quantiles']['measurement_x']['q1']
		q3_x = params['quantiles']['measurement_x']['q3']
		iqr_x = q3_x - q1_x

		q1_y = params['quantiles']['measurement_y']['q1']
		q3_y = params['quantiles']['measurement_y']['q3']
		iqr_y = q3_y - q1_y

		q1_z = params['quantiles']['measurement_z']['q1']
		q3_z = params['quantiles']['measurement_z']['q3']
		iqr_z = q3_z - q1_z
		
		# fetch the maximum sequence size for zero padding
		max_seq_size = params['max_seq_size']
		# load the feature scaler
		scaler = joblib.load('/home/gil/clew/scaler.dat')
		# load the prediction model
		model = load_model('/home/gil/clew/model.h5')

		# load the data
		data = pd.read_csv(dataFile)

		# keep last record in case of timestamp duplicates
		data.drop_duplicates(subset=["patient_id","timestamp"], keep='last', inplace=True)
		data = data[['patient_id','measurement_x','measurement_y','measurement_z']]
		# population outlier removal
		x_outlier_removed = data.measurement_x[(data.measurement_x < (q1_x-1.5*iqr_x)) | (data.measurement_x > (q3_x+1.5*iqr_x))]
		x_outlier_idx = x_outlier_removed.index
		y_outlier_removed = data.measurement_y[(data.measurement_y < (q1_y-1.5*iqr_y)) | (data.measurement_y > (q3_y+1.5*iqr_y))]
		y_outlier_idx = y_outlier_removed.index
		z_outlier_removed = data.measurement_z[(data.measurement_z < (q1_z-1.5*iqr_z)) | (data.measurement_z > (q3_z+1.5*iqr_z))]
		z_outlier_idx = z_outlier_removed.index
		outlier_removed = np.union1d(x_outlier_idx, y_outlier_idx)
		outlier_removed = np.union1d(outlier_removed, z_outlier_idx)
		data.drop(index=outlier_removed, inplace=True)
		data.sort_index(axis=0, inplace=True)

		# patient-specific outlier removal
		outlier_removed = np.array([])
		for pat_id in data.patient_id.unique():
			pat_df = data[data.patient_id == pat_id].copy()
			x_outlier_removed = remove_outlier_IQR(pat_df.measurement_x)[-1].index
			y_outlier_removed = remove_outlier_IQR(pat_df.measurement_y)[-1].index
			z_outlier_removed = remove_outlier_IQR(pat_df.measurement_z)[-1].index
			outlier_removed = np.union1d(outlier_removed, x_outlier_removed)
			outlier_removed = np.union1d(outlier_removed, y_outlier_removed)
			outlier_removed = np.union1d(outlier_removed, z_outlier_removed)
		data.drop(index=outlier_removed, inplace=True)
		data.sort_index(axis=0, inplace=True)

		measurements_label = ['measurement_x','measurement_y','measurement_z']
		## feature Min-Max rescaling
		data[measurements_label] = scaler.transform(data[measurements_label])

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
		labels = model.predict(X).reshape(-1).astype('int32').tolist()
		patients = data.patient_id.unique().tolist()
		prediction = {"predictions": {p:l for p,l in zip(patients, labels)}}

		# return the flask
		res = jsonify(prediction)

		return res

if __name__ == '__main__':
	app.run(debug=True, host='127.0.0.1', port=5000)