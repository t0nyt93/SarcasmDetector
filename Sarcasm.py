##Authors: Anthony Tyrrell and Kirk Stenett
# Oregon State University CS 331 Intro to AI
# Bayes Classifier for detecting sarcasm in tweets.
# Spring 2016

import re
import numpy
import time

from classes import sarClassm,fVec,Node,BayesClassifier
trainFile = "training_text.txt"
testFile = "test_text.txt"
test_print_file = "preprocessed_test.txt"
train_print_file = "preprocessed_train.txt"
results_print_file = "results.txt"

testClassm = sarClassm()
trainClassm = sarClassm()
trainFeatures = []
processed_sentences = []
testing_sentences = []
testFeatures = []

#@ param vals -a raw sentence straight from the file
#@ return - A tuple(lst,val) where lst is the list of all processed words and val is the sentences sarcasm value.
def trimSentence(Current_Sentence):
	processed_sentences = []
	processedWords = []
	s = Current_Sentence
	s = s.lstrip("(").rstrip(")")
	#Get the sentences sarcasm value
	isSarcastic = s[-3:]
	isSarcastic = isSarcastic[:-2]
	#More formatting
	s = s[:-4]
	s = s[-(len(s)-1):]
	s = s.rstrip("\"").rstrip(".")
	#ok you have the words, you need to SPLIT then VERIFY 
	words = s.split()
	for x in words:
		#---Conjoin apostrophes
		x = x.replace("\'","")
		#---(#'s and words')
		x = re.sub("[^A-Za-z0-9]", " ", x)
		#Deal with anycases where where words were seperated with symbols
		processedWords = x.split(" ")
		for y in processedWords:
			if y not in processed_sentences:
				processed_sentences.append(y)

	return (processed_sentences,isSarcastic)

def getVocabulary(filename,myClass):
	with open(trainFile,"r+") as inTrain:
		curLine = inTrain.readlines()
		for document in curLine:
			thisLine = trimSentence(document)
			for x in thisLine[0]:
				if x not in myClass.vocab:
					if x is not "":
						myClass.addWord(x)
		myClass.vocab = sorted(myClass.vocab)


def convertToFeatures(myClass,featureVectors,whichFile):
	M = myClass.size
	with open(whichFile,"r+") as myFile:
		doc = myFile.readlines()
		for sentence in doc:
			f = fVec(M)
			thisLine = trimSentence(sentence)
			#ok so now you have a list and a sarcasm value. MAKE A FEATURE VECTOR 
			#This vector references the classes vocab and fills out the list featureVector n 
			#Find the index of the vector
			vec = 0
			for x in thisLine[0]:
				vec+=1
				if x in myClass.vocab:
					wordsIndex = myClass.vocab.index(x)
					f.entries[wordsIndex] = 1
			f.entries[M] = thisLine[1]

			featureVectors.append(f)

def printFeats(myFile,listOfFeatures):

	iterData = iter(listOfFeatures)
	sentence = listOfFeatures[0]
	next(iterData)
	with open(myFile,"w+") as writeFile:
		for z in sentence:
			writeFile.write(str(z) + ",")
		writeFile.write("\n")
		for x in iterData:
			for y in x.entries:
				writeFile.write(str(y)+",")
			writeFile.write("\n")
				


def main():
	print ("Preprocessing engaged.")
	#PreProcessing ---------------------------------------------------
	#-------------------------------------------------------
	#---------------------------------------------
	#------------------------------------
	#1/2) Form the Vocabulary and strip punctuation
	getVocabulary(testFile,testClassm)
	getVocabulary(trainFile,trainClassm)

	#3)Convert TEST and TRAIN into a set of features
	convertToFeatures(trainClassm,trainFeatures,trainFile)
	convertToFeatures(testClassm,testFeatures,testFile)

	#Now you have all features of the data and the vocabularies
	#Make the heads of the lists vocab so we can print it prettily later.
	trainClassm.vocab.append("class label")	
	testClassm.vocab.append("class label")	

	trainFeatures.insert(0,trainClassm.vocab)
	testFeatures.insert(0,testClassm.vocab)
	#Print our Features to file.
	printFeats(test_print_file, testFeatures)
	printFeats(train_print_file,trainFeatures)
	
	#CLASSIFICATION STEP -------------------------------------------------------
	#-------------------------------------------------------------
	#------------------------------------------------
	#-------------------------------------
	#TRAIN ON TRAINING.TXT
	#Initialize our classifier with a list of features and vocabulary
	myClassifier = BayesClassifier(trainFeatures,trainClassm.vocab);

	#Calculate the base probability and calculates JPT for all words in initialized vocabulary
	myClassifier.CalcBaseProb()
	myClassifier.fillAndCompute()

	#Function returns a tuple, (#correct,#wrong,#total)
	print "Predicting training documents"
	results_training = myClassifier.predictFeatures(trainFeatures)
	tR = results_training[0]/(results_training[2])
	print "Predicting testing documents"
	results_testing = myClassifier.predictFeatures(testFeatures)
	tT = results_testing[0]/(results_testing[2])
	print "Writing results to file. Printing them as well..."
	time.sleep(1)
	print("Trained on TRAINING.txt, tested on TRAINING.txt... Predicted " + str(results_training[0]) + " correctly, " + str(results_training[1]) + " incorrectly,ate one.")
	print(str(results_training[0])+"/" + str(results_training[2]) + " = " + str(tR) + "%\ accuracy" )
	print("Training on TRAINING.txt, tested on TESTING.txt... Predicted " + str(results_testing[0]) + " correctly, " + str(results_testing[1]) + " incorrectly,ate one.")
	print(str(results_testing[0])+"/" + str(results_testing[2]) + " = " +  str(tT) + "%\ accuracy" )
	with open(results_print_file, "w+") as r:
		r.write("Trained on TRAINING.txt, tested on TRAINING.txt... Predicted " + str(results_training[0]) + " correctly, " + str(results_training[1]) + " incorrectly,ate one.\n")
		r.write(str(results_training[0])+"/" + str(results_training[2]) + " = " + str(tR) + "%\ accuracy\n" )
		r.write("Training on TRAINING.txt, tested on TESTING.txt... Predicted " + str(results_testing[0]) + " correctly, " + str(results_testing[1]) + " incorrectly,ate one.\n")
		r.write(str(results_testing[0])+"/" + str(results_testing[2]) + " = " + str(tT) + "%\ accuracy\n" )
	print "Results written to Results.txt"

if __name__ == '__main__':
	main()