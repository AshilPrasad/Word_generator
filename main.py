import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import  pickle


with open('9-RNN-LSTM/Tokenizer.pickle','rb') as handle:
    mytokenizer=pickle.load(handle)

model=tf.keras.models.load_model('9-RNN-LSTM/word_generator_model.h5')

input_text='prime'
predict_next_words=3

for i in range(predict_next_words):
    token_list=mytokenizer.texts_to_sequences([input_text])[0]
    token_list=pad_sequences([token_list],maxlen=model.input_shape[1],padding='pre')
    predicted=np.argmax(model.predict(token_list,),axis=-1)
    output_word=mytokenizer.index_word[predicted[0]]

    input_text+=" " + output_word
print(input_text)
