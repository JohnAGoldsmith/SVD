# coding = utf-16

import codecs
import os
import sys
import re
import math
import numpy as np
from  collections import defaultdict 
from os import path

#---------------------------------------------------------------------------------------#
#								                	#
#	This program creates a trigram file, bigram file, and a word list		#	
#	that is used by FindManifold.py				                        #
#	Begun by John Goldsmith and Wang Xiuli 2012.		           		#
#									                #
#---------------------------------------------------------------------------------------#

def mySVD(bigram_array, outfolder, rank ):

	U,Sigma,V = np.linalg.svd(bigram_array)       
	np.set_printoptions(precision = 2)

        print >>outfile_testing, "Bigrams", bigram_array
	print 
	print "U", U
	print "V", V
	print "Sigma", Sigma
	print "Sigma shape", Sigma.shape
 


        newpath = outfolder + "/eigenvectors/"
	
	try:
                os.mkdir(newpath) 
        except OSError:
	        print ("Creation of the directory %s failed" % newpath)

	outfile = open ( newpath + "testing.txt", "w")

	longest_length = 1
	for i in range(word_size):
            if len(index_to_word[i]) > longest_length:
                longest_length = len(index_to_word[i])

	for eigenvectornumber in range(rank):
            
            filename = newpath + str(eigenvectornumber)
            eigenfile = open (newpath + str(eigenvectornumber) + "_U_eigenvector.txt", "w")

	    eigenlist = list()
	    print  >>eigenfile, "eigenvector number", eigenvectornumber
	    for i in range(word_size):
		eigenlist.append((index_to_word[i], U[i,eigenvectornumber]))
	    eigenlist.sort(key = lambda x: x[1], reverse = True)
	    j = 0
	    for item in eigenlist:
		print  >>eigenfile, j, item [0], " "* (longest_length + 2 - len(item[0] )) ,  '{:f}'.format(item[1])
                j = j + 1
	    print >>eigenfile
	    eigenfile.close()
 
 	for eigenvectornumber in range(rank):
            
            filename = newpath + str(eigenvectornumber)
            eigenfile = open (newpath + str(eigenvectornumber) + "_V_eigenvector.txt", "w")

	    eigenlist = list()
	    print  >>eigenfile, "eigenvector number", eigenvectornumber
	    for i in range(word_size):
		eigenlist.append((index_to_word[i], V[eigenvectornumber,i]))
	    eigenlist.sort(key = lambda x: x[1], reverse = True)
	    j = 0
	    for item in eigenlist:
		print  >>eigenfile, j, item [0], " "* (longest_length + 2 - len(item[0] )) ,  '{:f}'.format(item[1])
                j = j + 1
	    print >>eigenfile
	    eigenfile.close()
 
 	outfile.close()
	return U,Sigma,V

def find_word_distances (U,Sigma,V, outfolder):
    width, height = U.shape
    VT = np.transpose(V)
    word_count = height
    distance_U = np.zeros((height,width))
    for this_word in range(10):
        neighbors = list()  
        print >>outfile_testing, "*1", index_to_word[this_word]      
        for that_word in range(10):
            distance = 0.0
            print >>outfile_testing,"*2\t", index_to_word[that_word], U[that_word,0]
            for column_no in range(width):
                diff = (U[this_word, column_no] - U[ that_word, column_no])
                delta =  diff**2.0 * Sigma[column_no]
	        distance += delta
                #distance 
                print >>outfile_testing, "*3\t\t word1: %10s  %8.5f word2: %10s %8.5f diff: %8.5f delta distance: %8.5f  distance: %8.5f singval %8.5f "%( index_to_word[this_word], U[this_word, column_no], index_to_word[that_word], U[ that_word, column_no], diff, delta, distance, Sigma[column_no])
	    #distance = math.sqrt(distance)
	    print >>outfile_testing, "\t",index_to_word[this_word], index_to_word[that_word], distance, "\n\n"

def read_sentence_list(sentence_list, bigrams, trigrams, word_to_index, contexts, contextfillers):
    i = 0
    total_bigram_count = 0.0
    for line in sentence_list:
        if i % 10000 == 0:
            print i / 10000,
        i += 1
	words = line.split()
	sentence_length = len(words)

        indices = list()
	for wordno in range(sentence_length):
		thisword = words[wordno]
                if thisword in word_to_index:
		    indices.append(word_to_index[thisword])
                else:
                    indices.append(-1)
	for wordno in range(sentence_length-1):
            if indices[wordno] == -1:
                continue
            elif indices[wordno+1] == -1:
                continue
            else: 
                bigrams[(indices[wordno],indices[wordno+1])] += 1
		total_bigram_count += 1
		if wordno + 2 < sentence_length:
			if indices[wordno+2] > 0:
		            trigrams[ (indices[wordno], indices[wordno+1], indices[wordno+2]) ] += 1
		            #print >> outfile_testing, 212, wordno, index_to_word[indices[wordno]], index_to_word[indices[wordno+1]], index_to_word[indices[wordno+2]], line
		            context = (indices[wordno], indices[wordno+2])
			    contexts[context ] += 1	
				
		            if context not in contextfillers:
		                    contextfillers[context] = dict()
		            if indices[wordno+1] not in contextfillers[context]:
		                    contextfillers[context][indices[wordno+1]] = 0
	 	            contextfillers[context][indices[wordno+1]] += 1

    return bigrams, trigrams, contexts, contextfillers, total_bigram_count         
#---------------------------------------------------------------------------#
#	Variables to be changed by user
#---------------------------------------------------------------------------#

language 		= "english-browncorpus"
#language                = "english-toydata"
language 		= "english-encarta"
#language 		= "german" 
unicode 		= False # True
FileEncoding 		= "ascii" #utf-16" #utf-16
filter_out_punctuation_flag = False    #if True, then all n-grams which contain a piece of punctuation is kept (same for contexts)
lower_all_capitals_flag = False
filter_words_by_frequency_flag = True
word_size = 1000
rank = 100
#---------------------------------------------------------------------------#
#	File names
#---------------------------------------------------------------------------#
	
infolder 		= '../../data/' + language + '/'
infilename 		= infolder + language +  ".txt"
infilename 		= infolder  +  "encarta.txt"
outfolder		= infolder + "ngrams/"
 
if not os.path.exists(outfolder):
	os.makedirs(outfolder)
	
#For output only:
suffix = ""
outfilename1 	= outfolder + language + suffix +  "_trigrams.txt"
outfilename2 	= outfolder + language + suffix +  "_alphabetized.trigrams.txt"
outfilename3 	= outfolder + language + suffix +  "_words.dx1"
outfilename4 	= outfolder + language + suffix +  "_bigrams.txt" 
outfilename5 	= outfolder + language + suffix +  "_alphabetized.bigrams.txt"
outfilename6 	= outfolder + language + suffix +  "_contexts.txt"
outfilename7	= outfolder + language + suffix +  "_context_fillers.txt"  
outfilename8    = outfolder + language + suffix +  "_length_sorted_corpus.txt"
outfilename9    = outfolder + language + suffix +  "_testing.txt"
if unicode:
        print "unicode"
	outfiletrigrams1 = codecs.open(outfilename1, "w", encoding = FileEncoding)
	outfiletrigrams2 = codecs.open(outfilename2, "w",encoding = FileEncoding)
	outfilewords 	 = codecs.open(outfilename3, "w",encoding = FileEncoding)
	file1 		 = codecs.open(infilename,encoding = FileEncoding)
	if SigTransformFlag:
		infile_sigtransforms = codecs.open(infile_sigtransforms_name, encoding=FileEncoding)
else:
	outfiletrigrams1 = open(outfilename1, "w")
	outfiletrigrams2 = open(outfilename2, "w")
	outfilewords 	 = open(outfilename3, "w")
	outfilebigrams4  = open(outfilename4, "w")
	outfilebigrams5  = open(outfilename5, "w")
	outfilecontexts  = open(outfilename6, "w")
	outfilecontextfillers = open(outfilename7, "w")
	outfile_length_sorted_corpus = open (outfilename8, "w")
        outfile_testing = open(outfilename9, "w")
	file1 = open(infilename)

print "Outfile: ", outfilename3

#--------------------------------filter_out_punctuation_flag-------------------------------------------#
#	Read file
#---------------------------------------------------------------------------#

SentenceLengthTarget = 5   # if we want only sentences of a particular length 
Punctuation = ".,;:!?()"
linenumber 		= 0
linesOfText 		= 0
numberOfWordTypes	= 0
numberOfWordTokens	= 0
myWords			= defaultdict(int)
trigrams 		= defaultdict(int)
bigrams			= defaultdict(int)
words			= defaultdict(int)
contexts 		= defaultdict(int)
contextfillers		= dict()
length_sorted_sentences = defaultdict(list)
sentence_list 		= list()
sep = "\t"

chosen_words = dict()


print "\n\nCreating bigram files. \nName of file being read:", infilename

for line in file1:
	wordno = 0
	if not line:
		continue
	linenumber += 1
	if linenumber%10000 == 0:
		print linenumber/10000,
	line = line[:-1]

        if filter_out_punctuation_flag:
		line = line.replace(".", " ")
		line = line.replace(",", " ")
		line = line.replace(";", " ")	
		line = line.replace("!", " ")	
		line = line.replace("?", " ")
		line = line.replace(":", " ")	
		line = line.replace(")", " ")	
		line = line.replace("(", " ")	
		line = line.replace("\\", " ")	
		#line = line.replace("\'", " ")	
		line = line.replace("\"", " ")	
        else:
		line = line.replace(".", " . ")
		line = line.replace(",", " , ")
		line = line.replace(";", " ; ")	
		line = line.replace("!", " ! ")	
		line = line.replace("?", " ? ")
		line = line.replace(":", " : ")	
		line = line.replace(")", " ) ")	
		line = line.replace("(", " ( ")
		#line = line.replace("\'", " \' ")	
                line = line.replace("\"", " \" ")	
		line = line.replace("\\", " \\  ")
        if lower_all_capitals_flag:	
            line = line.lower()
        sentence_list.append(line)
	words = line.split()
	sentence_length = len(words)
	if sentence_length not in length_sorted_sentences:
		length_sorted_sentences[sentence_length] = list()
	if sentence_length == SentenceLengthTarget:
	    print >>outfile_length_sorted_corpus, line
	length_sorted_sentences[sentence_length].append(words)


	buffer = list()
	linesOfText += 1
	longest_word_length = 5
	letter = ""

	for word in words:
		numberOfWordTokens += 1			
		myWords[word] += 1
		if len(word) > longest_word_length:
			longest_word_length = len(word)          

print "\n Read all words."
print "Number of sentences: ", len(sentence_list)
good_words = list()
word_to_index = dict()
index_to_word = dict()

print "size of high frequency list", word_size
#print >>outfile_testing, "Word frequencies"
total_word_count = 0.0
if filter_words_by_frequency_flag:

	word_count_list = list()
        word_count_list =  sorted(myWords.items(), key = lambda item: item[1], reverse=True)
        for i in range(word_size):
            if i < word_size:            
		this_word = word_count_list[i][0]
		good_words.append(this_word)
		word_to_index[this_word] = i
                index_to_word[i] = this_word
		total_word_count += word_count_list[i][1]
                #print >>outfile_testing, i, this_word

bigram_height = word_size
bigram_width = word_size
bigram_array = np.zeros([bigram_height,bigram_width])



#---------------------------------------------------------------------------#
#	read sentence list
#---------------------------------------------------------------------------#

bigrams, trigrams, contexts, contextfillers, total_bigram_count =  read_sentence_list(sentence_list, bigrams, trigrams, word_to_index, contexts, contextfillers)



print 
largestvalue = 0.0
for bigram in bigrams:
	word1 = index_to_word[bigram[0]]
	word2 = index_to_word[bigram[1]]
        count1 = float(myWords[word1])
        count2 = float(myWords[word2])
        bicount = bigrams[bigram]
        tbc = float(total_bigram_count)
        twc = float(total_word_count)
        freq1 = count1/twc
        freq2 = count2/twc
        bifreq = bicount/tbc
	if False:
		#  Mutual information:
		value = math.log( bifreq  /  (freq1 * freq2)  )
	elif False:
		#  Rectified mutual information:
		value = max( 0, math.log( bifreq /  (freq1 * freq2) ) ) 
	elif False:
		#  Weighted mutual information:
		value= bicount * math.log( ( bifreq)  /   (freq1 * freq2)  ) 
	elif True:
		#  Rectified weighted mutual information:
		value = max(0, bicount * math.log( bifreq / (  freq1 * freq2 ) ) ) 
	elif False:
		# Raw scores
		value = float( bicount )
        bigram_array[bigram[0],bigram[1]] = value
        print >>outfile_testing, "349    %15s %15s %6d  %8.5f %8.5f %8.5f %8.5f" %(  word1, word2,  bicount, freq1, freq2, bifreq,  bigram_array[bigram[0],bigram[1]] )
	if value > largestvalue:
             largestvalue = value
print >>outfile_testing, 353, "largest value", largestvalue
print "\nCompleted counting words, bigrams."



#---------------------------------------------------------------------------#
#	SVD
#---------------------------------------------------------------------------#

U,Sigma,V = mySVD(bigram_array, outfolder, rank)

print >>outfile_testing, bigram_array
print >>outfile_testing

print >>outfile_testing, Sigma.shape, "Sigma shape"
print >>outfile_testing, Sigma


find_word_distances(U,Sigma,V,outfolder)
 


#---------------------------------------------------------------------------#
#	Print output
#---------------------------------------------------------------------------#

numberOfWordTypes = len(myWords)

format_string_2 = "   {0:<20s} 	{1:>20d}" 
format_string_3 = "{0:>20s} {1:<20s} {2:8d}" 
format_string_4 = "{0:>20s} {1:<20s} {2:<20s} {3:8d}" 
format_string_5 = "{0:>13s} {1:>4s}{3:8d}" 
intro_string = "# data source: {0} \n# lines of text: {1}\n# language: {2} \n# number of word types: {3} \n# number of word tokens:{4}".format(infilename,linesOfText,language, numberOfWordTypes, numberOfWordTokens)
	
print >>outfilewords, intro_string
print >>outfiletrigrams1, intro_string 
print >>outfiletrigrams2, intro_string 
print >>outfilebigrams4, intro_string 
print >>outfilebigrams5, intro_string 
print >>outfilecontexts, intro_string 
#------------------------------------------#
#	sort bigrams, trigrams by frequency
#------------------------------------------#	

print "Frequency-sorted bigrams and trigrams."
 
topbigrams = [x for x in bigrams.iteritems()]
topbigrams.sort(key=lambda x:x[1], reverse=True)
	
for bigram in topbigrams:
	print >>outfilebigrams4, format_string_3.format(index_to_word[bigram[0][0]] ,index_to_word[bigram[0][1]],bigram[1])
 	#print >>outfilebigrams4, bigram[0], bigram[1]

toptrigrams = [x for x in trigrams.iteritems()]
toptrigrams.sort(key=lambda x:x[1], reverse=True)
	
for trigram in toptrigrams:
	print >>outfiletrigrams1,format_string_4.format( index_to_word[trigram[0][0]],index_to_word[trigram[0][1]],index_to_word[trigram[0][2]], trigram[1]) 
trigramcount = 0
 
topcontexts = [x for x in contexts.iteritems()]
topcontexts.sort(key=lambda x:x[1], reverse=True)

MINIMUM_NUMBER_OF_FILLERS =50	
for context in topcontexts:
	if  context[1]  < MINIMUM_NUMBER_OF_FILLERS:
		continue
        print >>outfilecontexts, index_to_word[context[0][0]],index_to_word[context[0][1]], context[1] 

for context in topcontexts:
	thisContext = context[0]
	if  context[1] < MINIMUM_NUMBER_OF_FILLERS:
		continue
 	print >>outfilecontextfillers, "\n", index_to_word[thisContext[0]] + " __ " + index_to_word[thisContext[1]]
	fillerList = sorted(contextfillers[thisContext].items(), key = lambda x: x[1], reverse = True )
	i = 0
	numcols = 5
	print >>outfilecontextfillers,"\t",
	for filler in fillerList:
		#print >>outfilecontextfillers,   filler
		print >>outfilecontextfillers, "{0:<13s} {1:<6d}".format(index_to_word[filler[0]], filler[1]),
		if i == numcols:
			print >>outfilecontextfillers, "\n\t", 
			i = 0
		else:
			i = i+1
	print >>outfilecontextfillers


#------------------------------------------#
#	alphabetize bigram, trigrams
#------------------------------------------#

print "Alphabetized bigrams and trigrams."

# ---- bigrams
bigrams_2 = [x for x in bigrams.iteritems()]
bigrams_2.sort(key=lambda x:x[0], reverse=False)
for bigram in bigrams_2:	 
	print >>outfilebigrams5, format_string_3.format(index_to_word[bigram[0][0]] , index_to_word[bigram[0][1]],bigram[1])

	#print >>outfilebigrams5, format_string_2.format(bigram[0], bigram[1])
 
# ---- trigrams: 
trigrams_2 = [x for x in trigrams.iteritems()]
trigrams_2.sort(key=lambda x:x[0], reverse=False)
	
for trigram in trigrams_2:
	print >>outfiletrigrams2,format_string_4.format( index_to_word[trigram[0][0]], index_to_word[trigram[0][1]],index_to_word[trigram[0][2]], trigram[1])	 
  
 




















 
 
#------------------------------------------#
#	sort and print words
#------------------------------------------#
	
top_words = [x for x in myWords.iteritems()]
top_words.sort(key=lambda x: x[1],reverse=True)
 
count = 0
for word in top_words:	 
	count += 1
	#print word
	print >>outfilewords, format_string_2.format(word[0],word[1])
#------------------------------------------#
#	finish
#------------------------------------------#

outfilewords.close()
outfilebigrams4.close()
outfilebigrams5.close()

outfiletrigrams1.close()
outfiletrigrams2.close()
outfilecontexts.close()
outfilecontextfillers.close()
