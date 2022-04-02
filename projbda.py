
# https://towardsdatascience.com/how-to-use-the-reddit-api-in-python-5e05ddfd1e5c
import requests
import json
from matplotlib import pyplot as plt
import matplotlib
import cv2
import os

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
from nltk.stem import PorterStemmer

from operator import itemgetter
nltk.download('stopwords')

def get_json(topic: str):
    # note that CLIENT_ID refers to 'personal use script' and SECRET_TOKEN to 'token'
    auth = requests.auth.HTTPBasicAuth('', '')

    # here we pass our login method (password), username, and password
    data = {'grant_type': 'password',
            'username': '',
            'password': ''}

    # setup our header info, which gives reddit a brief description of our app
    headers = {'User-Agent': 'MyBot/0.0.1'}

    # send our request for an OAuth token
    res = requests.post('https://www.reddit.com/api/v1/access_token',
                        auth=auth, data=data, headers=headers)

    # convert response to JSON and pull access_token value
    TOKEN = res.json()['access_token']

    # add authorization to our headers dictionary
    headers = {**headers, **{'Authorization': f"bearer {TOKEN}"}}

    # while the token is valid (~2 hours) we just add headers=headers to our requests
    requests.get('https://oauth.reddit.com/api/v1/me', headers=headers)

    res = requests.get("https://oauth.reddit.com/r/{}/".format(topic),
                       headers=headers, params={'limit': 100})
    return res.json()

import pandas as pd

subreddits = ['politics', 'sports', 'unpopularopinion', 'ipl', 'nature', 'gameofthrones',
              'harrypotter', 'doctorwho','australia','facebookwins','obama','TrumpCriticizesTrump']

def get_dataframe(topics_array):
    df = pd.DataFrame()
    for i in topics_array:
        data = get_json(i)
        for post in data['data']['children']:
            if post['data']['selftext'] == "":
                continue
            df = df.append({'selftext': post['data']['selftext']}, ignore_index=True)
    df.to_json('posts.json')
    return df


reddit_df = get_dataframe(subreddits)

def preprocessing(df):
    from nltk.corpus import stopwords
    nltk.download('stopwords')
    stop = stopwords.words('english')
    extra_stopwords = 'i', 'the', 'would', 'get', '-', 'even', '*', "it'", 'ani' "i'm", 'onli', 'seem', 'could', 'it’', 'i’m', 'also', 'realli', 'becaus', 'ha', 'peopl', 'hi', 'thi', 'wa'
    stop.extend(extra_stopwords)
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    df['lemmatized_words'] = df['selftext'].apply(
        lambda x: ' '.join([lemmatizer.lemmatize(w) for w in x.split()]))
    df['stemmed_words'] = df['lemmatized_words'].apply(lambda x: ' '.join([stemmer.stem(w) for w in x.split()]))

    df['without_stopwords'] = df['stemmed_words'].apply(
        lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    df['without_stopwords'].to_json('texts.json')
    return df['without_stopwords']

text=preprocessing(reddit_df)

def get_df(text):
  Df={}
  number_of_words=0
  for i in text:
    processed_tokens=i.split()
    for j in processed_tokens:
      Df[j]=0
      number_of_words+=len(processed_tokens)
  for i in text:
    processed_tokens=i.split()
    for j in processed_tokens:
      Df[j]+=1
  for i in Df.keys():
    Df[i]=Df[i]/number_of_words
  return Df
doc_frquency=get_df(text)
res = dict(sorted(doc_frquency.items(), key = itemgetter(1), reverse = True)[:20])
print(res)

image = cv2.imread(os.path.join('images', 'image1.jpeg'))

def convert_images_from_folder(folder):
    for file_name in os.listdir(folder):
        image = cv2.imread(os.path.join(folder, file_name))
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        file_name = file_name.replace(".jpg", "_gray")+".jpg"
        cv2.imwrite(os.path.join(folder, file_name), gray_image)
        plt.hist(gray_image.ravel(),256,[0,256])
        plt.savefig(folder+'/'+file_name.replace('.jpg','.svg'), format="svg")
convert_images_from_folder('images')
