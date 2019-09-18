from flask import Flask, render_template,url_for,request
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from tensorflow import keras
from spacy.lang.en import English
import spacy
import re
	

app = Flask(__name__)

@app.route('/')
def home():
 return render_template('home.html')


@app.route('/predict', methods = ['POST'])
def predict():
	new_model = tf.keras.models.load_model('my_model.h5')
	
	

	#A function for preprocessing 

	MAX_NB_WORDS = 50000
	# Max number of words in each complaint.
	MAX_SEQUENCE_LENGTH = 250
	# This is fixed.
	EMBEDDING_DIM = 100
	if request.method == 'POST':
	  comment = request.form['comment']
	text=pd.Series(comment)
	text=text.str.replace('\n','')
	text=text.str.replace('\r','')
	text=text.str.replace('\t','')
	  
	  #This removes unwanted texts
	text = text.apply(lambda x: re.sub(r'[0-9]','',x))
	text = text.apply(lambda x: re.sub(r'[/(){}\[\]\|@,;.:-]',' ',x))
	  
	  #Converting all upper case to lower case
	text= text.apply(lambda s:s.lower() if type(s) == str else s)
	  

	  #Remove un necessary white space
	text=text.str.replace('  ',' ')

	  #print(dataset.head(10))
	nlp=spacy.load("en_core_web_sm")
	text =text.apply(lambda x: ' '.join([word for word in x.split() if nlp.vocab[word].is_stop==False ]))

	tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
	tokenizer.fit_on_texts(text)
	word_index = tokenizer.word_index
#	print(len(word_index))

	X = tokenizer.texts_to_sequences(text)                         #Tokenize the dataset
	X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

	my_prediction = new_model.predict(X)
	#print(my_prediction)
	my_prediction=my_prediction[0][1]
	print(my_prediction)
	m=34
	return render_template('result.html', prediction =my_prediction)

if __name__ == '__main__':
 app.run(debug=True)