from flask import Flask, redirect, url_for, request,jsonify
import pickle
PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL

import numpy as np
import pandas as pd
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
df = pd.read_csv('../data/ballad.csv')


def get_title_to_idx():
   return pickle.load(open("../model/ballad_title_to_idx.pkl", 'rb'))

def get_cosine_tfidf():
   return pickle.load(open("../model/ballad_sims_cosine.pkl", 'rb'))

def get_vectorizer():
   return pickle.load(open("../model/count_vectorizer.pkl", 'rb'))

def get_model_lr():
   return pickle.load(open("../model/model_lr.pkl", "rb"))

def get_cosine_emotion():
   return pickle.load(open("../model/ballad_emotion_sims.pkl", "rb"))

title_to_idx = get_title_to_idx()
cosine_tfidf = get_cosine_tfidf()
vectorizer = get_vectorizer()
model_lr = get_model_lr()
emotion_sims = get_cosine_emotion()

def get_recommend_list(name):

   title_idx = title_to_idx[name]

   top_ten = np.argsort(cosine_tfidf[title_idx])[::-1][2:12]
   top_ten = list(map(str, top_ten))
   recommend_list = list(df.iloc[top_ten].title.values)
   return recommend_list, top_ten

def lyrics_genre_classification(lyrics):
   lyrics = [lyrics]
   vect = vectorizer.transform(lyrics)
   temp_df = pd.DataFrame(vect.A, columns=vectorizer.get_feature_names())
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
      emotion = -emotion_sims[int(cur_idx)]
      emotion_idx = np.argsort(emotion)[1:14]
      emotion_title = list(df.loc[emotion_idx, 'title'].values)
      print(emotion_title)
      print(type(emotion_title))
      cur_lyrics = df.loc[cur_idx,'lyrics']
      genre = lyrics_genre_classification(cur_lyrics)
      recommend_list, top_ten_list = map(list, get_recommend_list(request.form['name']))
      return jsonify({"recommend_list":recommend_list, 'genre' : genre, "recommend_index_list": top_ten_list, 'emotion_title': emotion_title})

if __name__ == '__main__':
   app.run(host='0.0.0.0', port=5000,debug = True)