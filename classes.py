##Authors: Anthony Tyrrell and Kirk Stenett
# Oregon State University CS 331 Intro to AI
# Bayes Classifier for detecting sarcasm in tweets.
# Spring 2016
# Classes.py contains all of the custom objects used to create this classifier.
import time

class sarClassm(object):
	"""docstring for sarClassm"""
	def __init__(self):
		super(sarClassm, self).__init__()
		self.vocab = []
		self.filename = ""
		self.size = 0

	def addWord(self,myWord):
		self.vocab.append(myWord)
		self.size+=1

	def sort(self):
		self.vocab = sorted(self.vocab)

class fVec(object):
	def __init__(self,arg):
		super(fVec,self).__init__()
		self.entries = [0] * (arg + 1)
	def getClassLabel(self):
		return self.entries[len(self.entries)-1]
	def getValueAtPosition(self,arg):
		return self.entries[arg]
	def isWordinVector(self,arg):
		if arg in self.entries:
			return True
		else:
			return False



class Node(object):
	def __init__(self,parentObject,initValue):
		self.values = {"TT":0.,"FF":0.,"TF":0.,"FT":0.}
		self.parent = parentObject
		self.word = initValue
	def CalcProbability(self,trainingData):
		#Set Self.values for all combinations of true and false
		#Training data is the feature vectors, we already have our initValue aka the word,
		#So we need to fill out the probabilities for values
		index, pos,cv,wordVal  = -1,-1,-1,-1
		tt,ff,tf,ft,totalT,totalF = 0.,0.,0.,0.,0.,0.
		classLabel = -1
		iterData = iter(trainingData)
		next(iterData)
		pos = self.parent.vocab.index(self.word)
		for feature in iterData:
			#1)see if the word is a 0 or 1:
			#2)see if the class label is a 0 or 1:
			wordVal = feature.getValueAtPosition(pos)
			cv = feature.getClassLabel()
			if (wordVal == 1) and (cv == "1"):
				#PRESENT AND SARCASTIC
				tt += 1.
				totalT += 1.
			elif (wordVal == 0) and (cv == "1"):
				#NOT PRESENT AND SARCASTIC
				ft += 1.
				totalT += 1.
			elif (wordVal == 1) and (cv == "0"):
				#PRESENT AND NOT SARCASTIC
				tf += 1.
				totalF += 1.
			elif (wordVal == 0) and (cv == "0"):
				#NOT PRESENT AND NOT SARCASTIC
				ff += 1.
				totalF += 1.

		#Calculating probability with uniform Dirichlet Priors.
		#N sub j is equal to 2 (1/0) for this
		self.values["TT"] = (tt + 1)/(totalT+ 2)
		self.values["FF"] = (ff + 1)/(totalF+ 2)
		self.values["FT"] = (ft + 1)/(totalT+ 2)
		self.values["TF"] = (tf + 1)/(totalF+ 2)

	def isProbValid(self):
		if self.probSum() == 1.0:
			return True
		else:
			return False
		#Ensure self.values adds up to one
	def probSum(self):
		#Returns the sum of the probability table, should be one.
		return self.values["TT"] + self.values["FF"] + self.values["FT"] + self.values["TF"]
	def queryProbability(self,presence,sarcasm):
		if (presence == 1) and (sarcasm == 1):
			return self.values["TT"]
		elif (presence == 1) and ( sarcasm == 0):
			return self.values["TF"]
		elif (presence == 0) and (sarcasm == 1):
			return self.values["FT"] 
		elif (presence == 0) and (sarcasm == 0):
			return self.values["FF"] 
		else:
			return -1 
	def PrintProbTable(self):
		print "--------------------------------------"
		for key in self.values:
			print "---" + key + " = " + str(self.values[key]) + "|"
		print "---" + "Sum = " + str(self.probSum()) + "|"
		print "--------------------------------------"


class BayesClassifier(object):
	def __init__(self,f,v):
		self.values = {"S":0.5,"NS":0.5}
		self.children = []
		self.tData = f
		self.vocab = v
	def CalcBaseProb(self):
		#Calculate the number of sarcastic and non sarcastic tweets.
		iterData = iter(self.tData)
		next(iterData)
		total = 0.
		vOne = 0.
		vZero = 0.
		for feat in iterData:
			total += 1.
			classValue = (feat.getClassLabel())
			if (classValue) is "1":
				vOne += 1.
			elif (classValue) is "0":
				vZero += 1.
			else:
				total -= 1.

		self.values["S"] = vOne / total
		self.values["NS"] = vZero / total
		print "Finished calculating base sarcasm probabilities"		
	
	def fillAndCompute(self):
		for x in self.vocab:
			self.children.append(Node(self,str(x)))

		for child in self.children:
			child.CalcProbability(self.tData)
		print "Classifier nodes filled with vocab and Joint Probability Tables are computed"
	def predictSentence(self,myVector,sentence):
		values = myVector.entries[:len(self.vocab)-1]
		actual_value = myVector.entries[len(self.vocab)-1:]
		#So we can use the data in self.children.values to help predict whether
		#or not this sentence is sarcastic. 
		predictedS = self.values["S"]
		predictedNS = self.values["NS"]
		pos = -1
		for x in range(0,2):
			index = 0
			for value in values:
				word = sentence[index]
				pos = self.vocab.index(word)
				#Let's see what the TF and TT are for this, get some sort of reduction going and return a 1/0
				if value is 1:
					#Word is present
					valProb = self.children[pos].queryProbability(value,x)
					
				elif value is 0:
					valProb = self.children[pos].queryProbability(value,x)
				if x is 0:
					predictedNS *= valProb
				else:
					predictedS *= valProb
				index +=1					
		if (predictedS > predictedNS):
			return 1
		elif (predictedS < predictedNS):
			return 0
		else:
			print "Error, cant predict."
	def predictFeatures(self,features):
		
		words_to_classify, results, correctResults = [],[],[]
		total = 0
		iterData = iter(features)
		#skip the first element in the list since it is a list() and not an fVec()
		sentence = features[0]
		next(iterData)
		for vector in iterData:
			value = vector.entries[len(vector.entries)-1]
			results.append(self.predictSentence(vector,sentence))
			correctResults.append(value)
			total  += 1

		#Results now stores your chronological predictions for the features given
		i = 0
		totCorrect = 0.
		totWrong = 0.
		for x in correctResults:
			if x.isdigit():
				if int(x) == results[i]:
					totCorrect+=1.
				else:
					totWrong+=1.
			else:
				total -= 1
			i += 1
		
		return (totCorrect,totWrong,total)







