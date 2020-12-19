#region		-> libraries

# built-in
import math
import time
import json
from random	import randint
from datetime	import datetime, timedelta
from warnings import filterwarnings
filterwarnings("ignore")	#	stop sklearn convergence error clogging output

# external
from requests	import get
import numpy as np
from sklearn.model_selection	import KFold
from sklearn.preprocessing	import PolynomialFeatures
from sklearn.linear_model	import LogisticRegression
from sklearn.neighbors	import KNeighborsClassifier
from sklearn.dummy	import DummyClassifier
from sklearn.metrics	import mean_squared_error, plot_roc_curve, plot_confusion_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d	import Axes3D

#endregion	-> libraries

#region		-> important values

MODEL_FORECAST_TIMEFRAME = 14	# model design
POSITIVE_WORDS = "(good|great|positive|buy)"	# implicit feature
SUBREDDITS = "bitcoin, cryptocurrency, cryptotrade, cryptomarkets"	# implicit feature
TIMEFRAMES = [1, 5, 10, 14, 30]	# feature
NUMS_COMMENTS = [10, 100, 500, 1000, 2000, 5000]	# feature

lr = None
lr_q = None
knn = None
knn_q = None

def scoring_algorithm(comment, positive_words) :
	
	weight = comment["score"]
	
	if weight <= 0 :
		
		return 0
		
	else :
		
		return weight
		

#endregion	-> important values

#region		-> APIs

# reddit => abstraction of the PushShift (reddit) API to search for comments indicating a positive/negative trend in a given cryptocurrency
#	@coin			: the name of the coin to search for mentions of in comments
#					:: string (eg: "bitcoin")
#	@cutoff_date	: the most recent date the search will consider when looking for comments
#					:: datetime object (eg: datetime.now())
#	@timeframe		: the number of days prior to the cutoff date that the search will consider when looking for comments
#					:: integer (eg: 14)
#	@subreddits		: the list of subreddits that will be searched for comments
#					:: comma-separated string (eg: "bitcoin, cryptomarket, cryptotrade")
#	@positive_words	: at least one word from this list must appear in a comment for it to be considered
#					:: logical string (eg: "(good|great|buy)", which corresponds to a search for "good" or "great" or "buy")
#	@num_comments	: the number of top-voted comments from the specified time period that will be considered
#					:: integer (eg: 1000)
#	returns			: a score for the query which will be used as training input for our model
#					:: integer
def reddit(coin, cutoff_date, timeframe, subreddits, positive_words, num_comments) :
	
	# implement the timeframe
	start_date = cutoff_date - timedelta(days = timeframe)
	
	# API docs at https://reddit-api.readthedocs.io/en/latest/ and https://github.com/pushshift/api/blob/master/README.md
	base_url = "https://api.pushshift.io/reddit/search/comment/?"
	
	# convert the datetime objects to epoch format requested by the API
	# ensure they are integers and converted to query-strings
	query_start_date = str(math.floor(start_date.timestamp()))
	query_cutoff_date = str(math.floor(cutoff_date.timestamp()))
	
	# initialise the query score
	tally = 0
	# enable retries if an error occurs
	keep_trying = True
	
	while (keep_trying == True) :
			
		try :
			
			# consult API docs for explanation of query fields
			r = get(
				base_url		+
				"q="			+	(coin + "+" + positive_words)	+	"&"	+
				"subreddit="	+	subreddits						+	"&"	+
				"size="			+	str(num_comments)				+	"&"	+
				"sort="			+	"desc"							+	"&"	+
				"sort_type="	+	"score"							+	"&"	+
				"after="		+	query_start_date				+	"&"	+
				"before="		+	query_cutoff_date				+	"&"	+
				"fields="		+	"score"
				, timeout = None
			)	
			
			# assign a score to each matched comment
			for comment in r.json()["data"] :
			
				# - - - - - - - - - - - - - - - - - - - - - - - - - - -
				# ALGORITHM TO DETERMINE OUR MODEL TRAINING INPUT DATA
				# - - - - - - - - - - - - - - - - - - - - - - - - - - -
				tally += scoring_algorithm(comment, positive_words)
				# - - - - - - - - - - - - - - - - - - - - - - - - - - -
				# /ALGORITHM TO DETERMINE OUR MODEL TRAINING INPUT DATA
				# - - - - - - - - - - - - - - - - - - - - - - - - - - -
			
			# query successful, return the query score
			keep_trying = False
			# consider generating multiple features from the comments
			return tally
	
		except :
			
			# error with query
			print("Error occurred: " + str(r.status_code))
			
			# 429 => rate limit being exceeded, risk of IP ban
			if (r.status_code == 429) :
				
				print("API asks that the program halts- exiting")
				keep_trying = False
				exit(0)
			
			# 502 => server slow to connect, no risk in retrying
			elif (r.status_code == 502) :
				
				print("API temporarily unavailable, attempting to reconnect...")
				# refresh the connection
				r.connection.close()
				time.sleep(3)


# CoinGecko => abstraction of the CoinGecko API to check if a given cryptocurrency increased or decreased in value over a given time period
#	@coin			: the name of the coin to compare the values for
#					:: string (eg: "bitcoin")
#	@cutoff_date	: the more recent date at which to value the coin to determine its growth/decline
#					:: datetime object (eg: datetime.now())
#	@timeframe		: the number of days after the cutoff date at which we're checking the value of the coin for comparison
#					:: integer (eg: 14)
#	returns			: whether the coin increased or decreased in value over the given time period
#					:: integer (1 = increased, 0 = decreased, -1 = nonexistent)
def CoinGecko(coin, cutoff_date, timeframe) :
	
	# get the date of the forecast
	forecast_date = cutoff_date + timedelta(days = timeframe)
	
	# API docs at https://www.coingecko.com/api/documentations/v3
	baseURL = "https://api.coingecko.com/api/v3"
	
	# convert the datetime objects to "dd-mm-yyyy" format requested by the API
	# ensure they are query-strings
	query_forecast_date = (str(forecast_date.day) + "-" + str(forecast_date.month) + "-" + str(forecast_date.year))
	query_cutoff_date = (str(cutoff_date.day) + "-" + str(cutoff_date.month) + "-" + str(cutoff_date.year))
	
	# enable retries if an error occurs
	keep_trying = True
	
	while (keep_trying == True) :
		
		try :
			
			# consult the API docs for explanation of query fields
			r_forecast = get(
				baseURL 		+ 
				"/coins/"		+	coin			+
				"/history?"		+
				"localization="	+	"false"			+	"&"	+
				"date="			+	query_forecast_date
				, timeout = None
			)
			
			# query successful
			keep_trying = False
		
		except :
			
			# error with query
			print("Error occurred: " + str(r_forecast.status_code))
			
			# 429 => rate limit being exceeded, risk of IP ban
			if (r_forecast.status_code == 429) :
				
				print("API asks that the program halts- exiting")
				keep_trying = False
				exit(0)
			
			# 502 => server slow to connect, no risk in retrying
			elif (r_forecast.status_code == 502) :
				
				print("API temporarily unavailable, attempting to reconnect...")
				# refresh the connection
				r_forecast.connection.close()
				time.sleep(3)
	
	# reset error status for the second query
	keep_trying = True

	while (keep_trying == True) :
		
		try :
			
			# consult the API docs for explanation of query fields
			r_cutoff = get(
				baseURL 		+ 
				"/coins/"		+	coin			+
				"/history?"		+
				"localization="	+	"false"			+	"&"	+
				"date="			+	query_cutoff_date
				, timeout = None
			)
			
			# query successful
			keep_trying = False
		
		except :
			
			# error with query
			print("Error occurred: " + str(r_cutoff.status_code))
			
			# 429 => rate limit being exceeded, risk of IP ban
			if (r_cutoff.status_code == 429) :
				
				print("API asks that the program halts- exiting")
				keep_trying = False
				exit(0)
			
			# 502 => server slow to connect, no risk in retrying
			elif (r_cutoff.status_code == 502) :
				
				print("API temporarily unavailable, attempting to reconnect...")
				# refresh the connection
				r_cutoff.connection.close()
				time.sleep(3)
	
	# if the coin didn't exist yet at the time of the cutoff date, an exception will be thrown
	try :
	
		# check if the given coin increased or decreased in value over the given timeframe, and return it as the training output
		if ((float(r_forecast.json()["market_data"]["current_price"]["eur"]) - float(r_cutoff.json()["market_data"]["current_price"]["eur"])) > 0) :
			
			return 1
			
		else :
			
			return 0
			
	# coin didn't exist, can't give a value for it
	except :
		
		return 0


#endregion	-> APIs

#region		-> training

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# ADJUST TRAINING PARAMETERS AND RUN TO GATHER MODEL TRAINING DATA
# REMEMBER ALSO THAT THE WEIGHTING ALGORITHM USED WITH THE REDDIT API IS ANOTHER DETERMINING FACTOR
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# fetch_training_data => gathers historical trends using the reddit and CoinGecko APIs to feed as training input/output to our model
def fetch_training_data(
	training_coins = ["bitcoin", "ethereum", "dogecoin", "litecoin", "tron"],
	training_cutoff_date_offsets = np.linspace(1, 2000, num = 150),
	feature_timeframes = TIMEFRAMES,
	training_subreddits = SUBREDDITS,
	training_positive_words = POSITIVE_WORDS,
	feature_nums_comments = NUMS_COMMENTS
) :
	
	# initialise training results
	training_data = []
	
	# use a selection of coins to hopefully learn general coin behaviour
	for coin in training_coins :
		
		# generate training data from snapshots of historical data
		for cutoff_date_offset in training_cutoff_date_offsets :
			
			# this is how far back in time we're taking our next snapshot from
			cutoff_date = (datetime.now() - timedelta(days = cutoff_date_offset))
			
			# compute the output of the next snapshot
			y = CoinGecko(coin, cutoff_date, MODEL_FORECAST_TIMEFRAME)
			
			# this snapshot labels the input features and output classifier with the associated coin & cutoff date considered
			snapshot = { "coin": coin, "cutoff_date": cutoff_date.isoformat(), "X": [], "y": y }
			
			# generate features based on given timeframes
			for timeframe in feature_timeframes :
				
				# generate features based on given numbers of comments
				for num_comments in feature_nums_comments :
					
					# compute the next feature of the snapshot
					snapshot["X"].append(reddit(coin, cutoff_date, timeframe, training_subreddits, training_positive_words, num_comments))
					print("feature retrieved: coin = " + coin + ", cutoff date = " + cutoff_date.isoformat() + ", num_comments = " + str(num_comments) + ", timeframe = " + str(timeframe))
					
					# wait to proceed to the next snapshot feature
					# PushShift API rate limit is 1 req/s
					time.sleep(1)
					
			# store and print the finalised snapshot
			training_data.append(snapshot)
			print(json.dumps(snapshot, indent = 4))
			
	# log the training data
	with open("dataset_" + str(datetime.now().isoformat()) + ".json", "w") as file :
		
		json.dump(training_data, file, indent = 4)
		
	return training_data


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# TRAIN MODELS BASED ON THE GATHERED TRAINING DATA AND OUTPUT RESULTS
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def model_training(
	local_data_file = None,
	num_kfold_splits = 5,
	test_qs = [1, 2, 3, 4, 5],
	test_Cs = [0.01, 0.1, 1.0, 10.0],
	test_ks = [1, 3, 5, 7, 9]
) :
	
	global lr
	global lr_q
	global knn
	global knn_q
	
	training_data = []
	
	if (local_data_file is None) :
		
		training_data = fetch_training_data()
		
	else :
		
		with open(local_data_file) as f :
			
			training_data = json.load(f)
				
	X = []
	y = []
	for item in training_data :
		
		X.append(item["X"])
		y.append(item["y"])
	
	X = np.array(X)
	y = np.array(y)
	
	kf = KFold(n_splits = num_kfold_splits)
	
	# logistic regression
	
	min_lr_err = -1
	min_lr_q = test_qs[0]
	min_lr_C = test_Cs[0]
	
	# initialise graphing of q vs C vs error
	lr_fig = plt.figure()
	lr_graph = Axes3D(lr_fig)
	
	for C_i in test_Cs :
		for q_i in test_qs :
			
			Xpoly = PolynomialFeatures(q_i).fit_transform(X)
		
			lr = LogisticRegression(C = C_i)
			
			lr_err = 0
			for train, test in kf.split(Xpoly) :
				
				lr.fit(Xpoly[train], y[train])
				
				lr_ypred = lr.predict(Xpoly[test])
				
				lr_err += mean_squared_error(y[test], lr_ypred)
				
			print("@ logistic regression : q = " + str(q_i) + ", C = " + str(C_i) + "; error = " + str(lr_err))
			
			if ((lr_err < min_lr_err) or (min_lr_err < 0)) :
				
				min_lr_q = q_i
				min_lr_C = C_i
				min_lr_err = lr_err
			
			# create a colour gradient in the z direction to better illustrate error differences
			# normalise to [0, 1] for an rgb value	
			lr_err_col = 1.0 if ((min_lr_err / lr_err) > 1.0) else (min_lr_err / lr_err)
			
			# plot the current point
			lr_graph.scatter(q_i, C_i, lr_err, color = (lr_err_col, (lr_err_col/2.0), (lr_err_col/3.0)))
			
	print("::: best logistic regression configuration : q = " + str(min_lr_q) + ", C = " + str(min_lr_C) + "; error of " + str(min_lr_err) + " :::")
	
	lr_q = min_lr_q
	lr_Xpoly = PolynomialFeatures(min_lr_q).fit_transform(X)
	lr = LogisticRegression(C = min_lr_C).fit(lr_Xpoly, y)
	
	# save the final logistic regression plot
	lr_graph.set_title("Logistic Regression: q vs C vs error")
	lr_graph.set_xlabel("q (hyperparameter degrees)")
	lr_graph.set_ylabel("C (L2 penalty cost parameter)")
	lr_graph.set_zlabel("mean squared error")
	plt.savefig("res/LogisticRegression_error_" + training_data[0]["cutoff_date"] + ".png", bbox_inches = "tight")
	plt.clf()
	
	# save the logistic regression ROC curve
	plot_roc_curve(lr, lr_Xpoly, y)
	plt.savefig("res/LogisticRegression_ROC_" + training_data[0]["cutoff_date"] + ".png", bbox_inches = "tight")
	plt.clf()
	
	# save the logistic regression confusion matrix
	plot_confusion_matrix(lr, lr_Xpoly, y)
	plt.savefig("res/LogisticRegression_ConfusionMatrix_" + training_data[0]["cutoff_date"] + ".png", bbox_inches = "tight")
	
	# kNN
	
	min_knn_err = -1
	min_knn_q = test_qs[0]
	min_knn_k = test_ks[0]
	
	# initialise graphing of q vs k vs error
	knn_fig = plt.figure()
	knn_graph = Axes3D(knn_fig)
	
	for k_i in test_ks :
		for q_i in test_qs :
			
			Xpoly = PolynomialFeatures(q_i).fit_transform(X)
		
			knn = KNeighborsClassifier(n_neighbors = k_i)
			
			knn_err = 0
			for train, test in kf.split(Xpoly) :
				
				knn.fit(Xpoly[train], y[train])
				
				knn_ypred = knn.predict(Xpoly[test])
				
				knn_err += mean_squared_error(y[test], knn_ypred)
				
			print("@ kNN : q = " + str(q_i) + ", k = " + str(k_i) + "; error = " + str(knn_err))
				
			if ((knn_err < min_knn_err) or (min_knn_err < 0)) :
				
				min_knn_q = q_i
				min_knn_k = k_i
				min_knn_err = knn_err
				
			# create a colour gradient in the z direction to better illustrate error differences	
			# normalise to [0, 1] for an rgb value
			knn_err_col = 1.0 if ((min_knn_err / knn_err) > 1.0) else (min_knn_err / knn_err)
			
			# plot the current point
			knn_graph.scatter(q_i, k_i, knn_err, color = (knn_err_col, (knn_err_col/2.0), (knn_err_col/3.0)))
			
	print("::: best kNN configuration : q = " + str(min_knn_q) + ", k = " + str(min_knn_k) + "; error of " + str(min_knn_err) + " :::")
	
	knn_q = min_knn_q
	knn_Xpoly = PolynomialFeatures(min_knn_q).fit_transform(X)
	knn = KNeighborsClassifier(n_neighbors = min_knn_k).fit(knn_Xpoly, y)
	
	# save the final kNN plot
	knn_graph.set_title("kNN: q vs k vs error")
	knn_graph.set_xlabel("q (hyperparameter degrees)")
	knn_graph.set_ylabel("k (number of checked neighbours")
	knn_graph.set_zlabel("mean squared error")
	plt.savefig("res/kNN_error_" + training_data[0]["cutoff_date"] + ".png", bbox_inches = "tight")
	plt.clf()
	
	# save the kNN ROC curve
	plot_roc_curve(knn, knn_Xpoly, y)
	plt.savefig("res/kNN_ROC_" + training_data[0]["cutoff_date"] + ".png", bbox_inches = "tight")
	plt.clf()
	
	# save the kNN confusion matrix
	plot_confusion_matrix(knn, knn_Xpoly, y)
	plt.savefig("res/kNN_ConfusionMatrix_" + training_data[0]["cutoff_date"] + ".png", bbox_inches = "tight")
	
	#dummy classifier
	
	dummy = DummyClassifier()
	dummy_err = 0
	
	for train, test in kf.split(Xpoly) :
		
		dummy.fit(Xpoly[train], y[train])
		
		dummy_ypred = dummy.predict(Xpoly[test])
		
		dummy_err += mean_squared_error(y[test], dummy_ypred)
		
	print("@ dummy classifier :  error = " + str(dummy_err))
	
	# save the dummy ROC curve
	plot_roc_curve(dummy, Xpoly, y)
	plt.savefig("res/dummy_ROC_" + training_data[0]["cutoff_date"] + ".png", bbox_inches = "tight")
	plt.clf()
	
	# save the dummy confusion matrix
	plot_confusion_matrix(dummy, Xpoly, y)
	plt.savefig("res/dummy_ConfusionMatrix_" + training_data[0]["cutoff_date"] + ".png", bbox_inches = "tight")


def prepare_model() :
	
	start_time = time.time()
	print("-------- BEGINNING TRAINING --------")
	model_training(local_data_file = "res/dataset_2020-12-19T18:20:38.225562.json")
	print("-------- TRAINING COMPLETE --------")
	print("-> completed in " + str(time.time() - start_time) + " seconds")


#endregion	-> training

#region		-> application

def predict(coin) :
	
	cutoff_date = datetime.now() - timedelta(days = 1)
	
	snapshot = { "coin": coin, "cutoff_date": cutoff_date.isoformat(), "X": [], "y": [] }
	
	for timeframe in TIMEFRAMES :
		
		for num_comments in NUMS_COMMENTS :
			
			# compute the next feature of the snapshot
			snapshot["X"].append(reddit(coin, cutoff_date, timeframe, SUBREDDITS, POSITIVE_WORDS, num_comments))
			print("feature retrieved: num_comments = " + str(num_comments) + ", timeframe = " + str(timeframe))
			
			# wait to proceed to the next snapshot feature
			# PushShift API rate limit is 1 req/s
			time.sleep(1)
	
	print("LogisticRegression prediction: " + str(lr.predict(PolynomialFeatures(lr_q).fit_transform([snapshot["X"]]))))
	print("kNN prediction: " + str(knn.predict(PolynomialFeatures(knn_q).fit_transform([snapshot["X"]]))))
	

#endregion	-> product

#region		-> main

def main() :
	
	prepare_model()
	
	predict("tether")


main()

#endregion	-> main
