import sys
import pandas as pd
import numpy as np
from gensim.models import FastText
import tensorflow as tf
from termcolor import colored, cprint

class Text_Classification:
  def __init__(self):
    self.stop_words = pd.read_csv('Dataset/bangla_stop_word_471.csv')
    self.fasttext_model = FastText.load('Pretrained_Models/fasttext_model_20/fasttext_model.model')
    self.Text_Classifier_model = tf.keras.models.load_model('Saved_Model/Biderectional.h5')

  def change_model(self, num):
    if num == 2: self.Text_Classifier_model = tf.keras.models.load_model('Saved_Model/Biderectional_alt.h5') ;return 'Bi-LSTM_V2'
    if num == 1: self.Text_Classifier_model = tf.keras.models.load_model('Saved_Model/Biderectional.h5') ;return 'Bi-LSTM_V1'

  def find_class(self, text_vector):
    classes = ['economy', 'education', 'entertainment', 'international', 'sports','state', 'technology']
    text_arr = np.asarray(text_vector)
    output_values = list(self.Text_Classifier_model.predict(text_arr)[0])
    return classes[output_values.index(max(output_values))]

  def input_preprocessing(self, input_text):
    input_text = [word for word in input_text.split() if not word in self.stop_words]
    if len(input_text) > 350:
      return 'Please enter 350 word as input'

    text_vector = []
    word_to_vec_len = 350                                           
    k = len(input_text)
    temp = []
    for j in range(k):
      try:
        vector_a = self.fasttext_model.wv.get_vector(input_text[j])
        temp.append(vector_a)
      except:
        temp.append(np.zeros((20,), dtype=np.float32))
        continue
    if k < word_to_vec_len:                                           
      k = word_to_vec_len - k;
      for p in range(k):
        temp.append(np.zeros((20,), dtype=np.float32))
    text_vector.append(temp)
    
    return self.find_class(text_vector).upper()

if __name__ == "__main__":
  Classification = Text_Classification()
  cprint("           Text Classifier Tool Options", 'cyan', attrs=['bold'], file=sys.stderr)
  def option_info(model_v):
    print(colored("=====================================================", 'yellow'))
    print("1) -1 Exit from the System\n2) 1 for Bi-LSTM_V1 to Activate\n2) 2 for Bi-LSTM_V2 to Activate")
    print(colored("=====================================================", 'yellow'))
    print(colored("=====================================================", 'green'))
    print("               Model Arc. : ",colored(model_v, 'red'))
    print(colored("                 [Start Classify]", 'yellow'))
    print(colored("=====================================================", 'green'))
  option_info('Bi-LSTM_V1')
  while 1:
    input_text = input('Input Text (str) : ')
    if input_text == '-1': 
      print(colored("=====================================================", 'yellow'))
      print(colored('                !!! Thaknk You !!! ','green')); 
      print(colored("=====================================================", 'yellow'))
      break

    if input_text == '1': 
      option_info(Classification.change_model(int(input_text)))
    elif input_text == '2': 
      option_info(Classification.change_model(int(input_text)))
    else:
      output_class = Classification.input_preprocessing(input_text)
      print(colored(input_text, 'blue'))
      print('                  Class is :',colored(output_class, 'red'))
      print(colored("=====================================================", 'green'))