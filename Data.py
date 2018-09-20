import os,sys
import json

class Data:

	def __init__ (self,structure, tweet_data, source):
		self.structure = structure
		self.tweet_data = tweet_data
		self.source = source





path = './semeval2017-task8-dataset/rumoureval-data/charliehebdo'

lis = []
fold = os.listdir(path)
for i in fold:
	if (i == '.DS_Store'):
		continue
	temp_source = path + '/' + i + '/source-tweet/'
	temp_replies = path + '/' + i + '/replies/'
	temp_struct = path + '/' + i 
	with open(temp_struct + '/structure.json') as f:
		structure = json.load(f)
	source_file = os.listdir(temp_source)
	source = source_file[0].split('.')[0]
	tweet_data = {}	
	with open(temp_source + source_file[0]) as f:
		tweet_data[source] = (json.load(f))

	reply_file = os.listdir(temp_replies)
	for j in reply_file:
		with open(temp_replies + j) as f:
			tweet_data[j.split('.')[0]] = (json.load(f))

	lis.append(Data(structure, tweet_data, source))

for i in lis:
	print i.source
