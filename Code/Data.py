import os,sys
import json
from gensim.models import Word2Vec
import nltk
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
import string
from nltk.tokenize import RegexpTokenizer
from textblob.classifiers import NaiveBayesClassifier
import pickle
import re
from textblob import TextBlob
import csv

filename = 'GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(filename, binary=True, limit=100000)
# f = open("model.pkl", "w")
# pickle.dump(model, f)
# f.close()
stop = set(nltk.corpus.stopwords.words('english'))
print "done"


class Data:

	def __init__ (self,structure, tweet_data, source):
		self.structure = structure
		self.tweet_data = tweet_data
		self.source = source

	def getfeature(self, tweet):
		text = tweet["text"]
		feature = []
		words = nltk.word_tokenize(text)

		tokenizer = RegexpTokenizer(r'\w+')
		word_nopunc = tokenizer.tokenize(text)
		word_nopunc = [i for i in  word_nopunc if i not in stop]

		# top 20 features using word2vec
		for i in word_nopunc:
			if i in model.wv:
				feat_list = model.wv[i].tolist()
				feature.extend(feat_list[:20])

		#append 0 if no feature found
		if (len(feature) < 100):
			for i in range(len(feature),101):
				feature.append(0)
		feature = feature[:100]

		# Has question marks
		if text.find('?') > 0:
			feature.append(1)
		else:
			feature.append(0)

		# has ! 
		if text.find('!') > 0:
			feature.append(1)
		else:
			feature.append(0)	
		
		# has hastag
		if (len(tweet['entities']['hashtags']) > 0):
			# feature.append(len(tweet['entities']['hashtags']))
			feature.append(1)
		else:
			feature.append(0) 

		# has usermention
		if (len(tweet['entities']['user_mentions']) > 0):
			# feature.append(len(tweet['entities']['user_mentions']))
			feature.append(1)
		else:
			feature.append(0)
				
		# has url
		if (len(tweet['entities']['urls']) > 0):
			# feature.append(len(tweet['entities']['urls']))
			feature.append(1)
		else:
			feature.append(0)	

		# has media
		if ('media' in tweet['entities']):
			# feature.append(len(tweet['entities']['media']))
			feature.append(1)
		else:
			feature.append(0)

		# sentiment analysis
		clean_tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())
		analysis = TextBlob(clean_tweet)

		if analysis.sentiment.polarity > 0:
			feature.append(1)
		else:
			feature.append(0)

		# # has poll
		# if ('polls' in tweet['entities']):
		# 	# feature.append(len(tweet['entities']['media']))
		# 	feature.append(1)
		# else:
			# feature.append(0)

		# Likes
		# if ((tweet['favorite_count']) > 0):
		# 	# feature.append(len(tweet['entities']['media']))
		# 	feature.append((tweet['favorite_count']))
		# else:
		# 	feature.append(0)

		# # Retweets
		# if ((tweet['retweet_count']) > 0):
		# 	# feature.append(len(tweet['entities']['media']))
		# 	feature.append((tweet['retweet_count']))
		# else:
		# 	feature.append(0)	

		#	favourited
		# if ('favourited' in tweet and tweet['favourited']):
		# 	feature.append(1)
		# else:
		# 	feature.append(0)	 		

		# #	Retweeted
		# if ('retweeted' in tweet and tweet['retweeted']):
		# 	feature.append(1)
		# else:
		# 	feature.append(0)	
		# is source
		# if (source == )

		# Capital to lower case ratio
		uppers = [l for l in text if l.isupper()]
		capitalratio = len(uppers)/len(text)	
		feature.append(capitalratio)

		count_punct = 0
		# negative words list 
		neg_words = ["not", "no", "nobody", "none", "never", "neither", "nor", "nowhere", "hardly", "scarcely", "barely", "don't", "isn't", "wasn't", "shouldn't", "wouldn't", "couldn't", "doesn't"]

		count_neg_words = 0
		# count number of punctuations and negative words
		for i in words:
			if (i in (string.punctuation)):
				count_punct += 1
			if (i in neg_words):
				count_neg_words += 1
		
		feature.append(count_punct)
		feature.append(count_neg_words)
		swearwords = []
		with open('badwords.txt', 'r') as f:
			for line in f:
				swearwords.append(line.strip().lower())

		hasswearwords = 0
		for token in word_nopunc:
			if token in swearwords:
				hasswearwords += 1
		feature.append(hasswearwords)


		return feature

	def extract_features(self):
		feat_dict = {}
		for i in self.tweet_data:
			feat_dict[i] = self.getfeature(self.tweet_data[i])
		# print len(feat_dict)
		return feat_dict	






# print "done"
path = '../Dataset/semeval2017-task8-dataset/rumoureval-data'

data = []
fold = os.listdir(path)

# Read DATA

for k in fold:
	if (k == '.DS_Store'):
		continue
	temp_files = path + '/' + k
	lis = []
	temp_inner = os.listdir(temp_files)

	# Get data for each topic 
	for i in temp_inner:
		if (i == '.DS_Store' or i == "."):
			continue
		temp_source = temp_files + '/' + i + '/source-tweet/'
		temp_replies = temp_files + '/' + i + '/replies/'
		temp_struct = temp_files + '/' + i 

		# store structure of tweets
		with open(temp_struct + '/structure.json') as f:
			structure = json.load(f)

		# store source tweet
		source_file = os.listdir(temp_source)
		source = source_file[0].split('.')[0]

		# store all twitter data
		tweet_data = {}	
		with open(temp_source + source_file[0]) as f:
			tweet_data[source] = (json.load(f))

		reply_file = os.listdir(temp_replies)
		for j in reply_file:
			with open(temp_replies + j) as f:
				tweet_data[j.split('.')[0]] = (json.load(f))

		lis.append(Data(structure, tweet_data, source))
	data.append(lis)

with open('../Dataset/las_vegas_shootout.json') as f:
	line = f.readline()
	t_d= {}
	while line:
		temp_t = json.loads(line.strip())
		t_d[temp_t['id']] = temp_t
		line = f.readline()
	lis.append(Data(None, t_d, None))
data.append(lis)

with open('../Dataset/tweets_california_shootout.json') as f:
	line = f.readline()
	t_d= {}
	while line:
		temp_t = json.loads(line.strip())
		t_d[temp_t['id']] = temp_t
		line = f.readline()
	lis.append(Data(None, t_d, None))
data.append(lis)


# Find feature vectors for each tweet
X_data = {}
for i in data:
	for j in i:
		X_data = dict(X_data.items() + j.extract_features().items())

# get training labels
X_label = {}
path = '../Dataset/semeval2017-task8-dataset/traindev/rumoureval-subtaskA-train.json'
with open(path) as f:
	X_label = json.load(f)

with open('../Dataset/newdata.csv') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	line_count = 0
	for row in csv_reader:
		if (line_count == 0):
			continue
		X_label[str(row[-3])] = str(row[-1])


# Get Test data
path = '../Dataset/semeval2017-task8-test-data/'
fold = os.listdir(path)
test_data = []


for i in fold:
	if (i == '.DS_Store' or i == "."):
		continue
	temp_source = path + '/' + i + '/source-tweet/'
	temp_replies = path + '/' + i + '/replies/'
	temp_struct = path + '/' + i 

	# store structure of tweets
	with open(temp_struct + '/structure.json') as f:
		structure = json.load(f)

	# store source tweet
	source_file = os.listdir(temp_source)
	source = source_file[0].split('.')[0]

	# store all twitter data
	tweet_data = {}	
	with open(temp_source + source_file[0]) as f:
		tweet_data[source] = (json.load(f))

	reply_file = os.listdir(temp_replies)
	for j in reply_file:
		with open(temp_replies + j) as f:
			tweet_data[j.split('.')[0]] = (json.load(f))

	test_data.append(Data(structure, tweet_data, source))

# get testing features
Y_data = {}
for i in test_data:
	Y_data = dict(Y_data.items() + i.extract_features().items())

# get testing labels
Y_label = {}
path = '../Dataset/test_label.json'
with open(path) as f:
	Y_label = json.load(f)


f = open("training.pkl", "w")
pickle.dump((X_data, X_label), f)
f.close()

f = open("testing.pkl", "w")
pickle.dump((Y_data, Y_label), f)
f.close()