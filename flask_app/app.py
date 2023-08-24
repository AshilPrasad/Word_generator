from flask import Flask,render_template,request,jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import  pickle

app=Flask(__name__)

with open('9-RNN-LSTM/Tokenizer.pickle','rb') as handle:
    mytokenizer=pickle.load(handle)

model=tf.keras.models.load_model('9-RNN-LSTM/word_generator_model.h5')


@app.route('/')

def index():
    return render_template('index.html')

@app.route('/predict',methods=["POST"])

def predict():
    input_text=request.form['input_Text']  #key name
    num_words=int(request.form['num_words']) #our input is text form we need to put int 

    generated_text=input_text

    for i in range(num_words):
         token_list=mytokenizer.texts_to_sequences([generated_text])[0]
         token_list=pad_sequences([token_list],maxlen=model.input_shape[1],padding='pre')
         predicted=np.argmax(model.predict(token_list,),axis=-1)
         output_word=mytokenizer.index_word[predicted[0]]

         generated_text+=" " + output_word

    return jsonify({"generated_text":generated_text})

if __name__== '__main__':
    app.run(debug=True)