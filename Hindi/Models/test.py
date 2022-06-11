import requests
import pandas as pd
import json
from tqdm import tqdm


df = pd.read_csv('lemmatized_news.csv')

""" baggage_handler = {}

news_to_send = df['News'].tolist()
print(len(news_to_send))
label_to_send = df['Label'].tolist()
print(len(label_to_send))
baggage_handler['News'] = news_to_send
baggage_handler['Label'] = label_to_send
print(len(baggage_handler['News']))
print(len(baggage_handler['Label']))

# print(json.dumps(baggage_handler, indent=4))
response = requests.post('http://127.0.0.1:5000/allNews', data=baggage_handler)
print("RESPONCE TXT: ", response.json()) """

""" for i in tqdm(df.index):
    response = requests.post(
        'http://127.0.0.1:5000/allNews', data={'News': df['News'][i]})
    print(response.json())
    print() """

print(df['News'][0])
print(df['Label'][0])
