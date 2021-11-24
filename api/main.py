from flask import Flask, redirect, url_for, request,jsonify
import pickle
PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL

import numpy as np
import pandas as pd
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
df = pd.read_csv('../data/ballad.csv', encoding='cp949')


def get_title_to_idx():
   return pickle.load(open("../model/ballad_title_to_idx.pkl", 'rb'))

def get_cosine_tfidf():
   return pickle.load(open("../model/ballad_sims_cosine_tfidf.pkl", 'rb'))

def get_vectorizer():
   return pickle.load(open("../model/count_vectorizer.pkl", 'rb'))

def get_model_lr():
   return pickle.load(open("../model/model_lr.pkl", "rb"))

title_to_idx = get_title_to_idx()
cosine_tfidf = get_cosine_tfidf()
vectorizer = get_vectorizer()
model_lr = get_model_lr()

def get_recommend_list(name):

   title_idx = title_to_idx[name]
   #print(f'current index : {title_idx}')

   top_ten = np.argsort(cosine_tfidf[title_idx])[::-1][1:11]
   #print(top_ten)
   top_ten = list(map(str, top_ten))
   recommend_list = list(df.iloc[top_ten].title.values)
   return recommend_list, top_ten
   #return jsonify({"recommend_list": list(df.iloc[top_ten].title.values)})


def lyrics_genre_classification(lyrics):
   lyrics = [lyrics]
   vect = vectorizer.transform(lyrics)
   temp_df = pd.DataFrame(vect.A, columns=vectorizer.get_feature_names_out())
   temp_result = list(model_lr.predict(temp_df))
   if temp_result[0] == 1:
      return 'ballad, broke up'
   else:
      return 'ballad'

@app.route('/recommend', methods=['POST'])
def hello_world():
   if request.method == 'POST':
      name = request.form['name']
      cur_idx = list(df[df['title'] == name].index)[0]
      print(cur_idx)
      cur_lyrics = df.loc[cur_idx,'lyrics']
      print(cur_lyrics)
      genre = lyrics_genre_classification(cur_lyrics)
      print(genre)
      recommend_list, top_ten_list = map(list, get_recommend_list(request.form['name']))
      print(top_ten_list)
      return jsonify({"recommend_list":recommend_list, 'genre' : genre, "recommend_index_list": top_ten_list})

if __name__ == '__main__':
   app.run(host='0.0.0.0', port=5000,debug = True)