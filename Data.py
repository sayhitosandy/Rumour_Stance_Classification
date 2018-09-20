import os,sys
import json
import pickle
import nltk
import string
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

filename = 'GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(filename, binary=True, limit=100000)
stop = set(nltk.corpus.stopwords.words('english'))
print ("done")


class Data:

	def __init__ (self,structure, tweet_data, source):
		self.structure = structure
		self.tweet_data = tweet_data
		self.source = source

	def getfeature(self, text):
		feature = []
		words = nltk.word_tokenize(text)
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
		feature.append(count_punct)
		feature.append(count_neg_words)

		return feature

	def extract_features(self):
		feat_dict = {}
		for i in self.tweet_data:
			feat_dict[i] = self.getfeature(self.tweet_data[i]['text'])
		# print len(feat_dict)
		return feat_dict	






# print "done"
path = './semeval2017-task8-dataset/rumoureval-data'

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

# print(data)

# Find feature vectors for each tweet
X_data = {}
# print(type(X_data))
for i in data:
	for j in i:
		# X_data = dict(X_data.items() + j.extract_features().items())
		# print (type(j.extract_features()))
		X_data.update(j.extract_features())
		
# get training labels
X_label = {}
path = './semeval2017-task8-dataset/traindev/rumoureval-subtaskA-train.json'
with open(path) as f:
	X_label = json.load(f)



# Get Test data
path = './semeval2017-task8-test-data/'
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
	# Y_data = dict(Y_data.items() + i.extract_features().items())
	Y_data.update(i.extract_features())

# get testing labels
Y_label = {}
path = './test_label.json'
with open(path) as f:
	Y_label = json.load(f)


f = open("training.pkl", "wb")
pickle.dump((X_data, X_label), f)
f.close()

f = open("testing.pkl", "wb")
pickle.dump((Y_data, Y_label), f)
f.close()



