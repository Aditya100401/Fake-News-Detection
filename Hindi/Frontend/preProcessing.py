from indicnlp.tokenize import indic_tokenize
from numpy import unicode
import pandas as pd
import stanza
nlp = stanza.Pipeline(lang="hi", processors="tokenize,pos,lemma")


def lemmatization(data):
    hindi_doc = nlp(data)
    lemmas = [word.lemma for sent in hindi_doc.sentences for word in sent.words]

    return ' '.join(i for i in lemmas if i != None)


def extractStop():
    stop = open('final_stopwords.txt',
                encoding='utf-8', errors='ignore')
    stopwords = []
    for x in stop:
        x = x.replace('\n', '')
        stopwords.append(x)

    return stopwords


def clean_stopwords(indic_string, stopwords):
    str_temp = ""
    rem = []
    for words in indic_string:
        if unicode(words) not in stopwords:
            str_temp += words
            str_temp += " "
            rem.append(str_temp)
        str_temp = ""
    return rem


def tokenization(indic_string):
    tokens = []
    for t in indic_tokenize.trivial_tokenize(indic_string):
        tokens.append(t)
    return tokens


def stemming(data):
    suffixes = {
        1: ["ो", "े", "ू", "ु", "ी", "ि", "ा"],
        2: ["कर", "ाओ", "िए", "ाई", "ाए", "ने", "नी", "ना", "ते", "ीं", "ती", "ता", "ाँ", "ां", "ों", "ें"],
        3: ["ाकर", "ाइए", "ाईं", "ाया", "ेगी", "ेगा", "ोगी", "ोगे", "ाने", "ाना", "ाते", "ाती", "ाता", "तीं", "ाओं", "ाएं", "ुओं", "ुएं", "ुआं"],
        4: ["ाएगी", "ाएगा", "ाओगी", "ाओगे", "एंगी", "ेंगी", "एंगे", "ेंगे", "ूंगी", "ूंगा", "ातीं", "नाओं", "नाएं", "ताओं", "ताएं", "ियाँ", "ियों", "ियां"],
        5: ["ाएंगी", "ाएंगे", "ाऊंगी", "ाऊंगा", "ाइयाँ", "ाइयों", "ाइयां"],
    }
    stems = []
    for word in data:
        word = word.strip()
        for L in 5, 4, 3, 2, 1:
            if len(word) > L + 1:
                for suffix in suffixes[L]:
                    if word.endswith(suffix):
                        word = word[:-L]
        stems.append(word)
    return ' '.join(i for i in stems)


def Preprocesing(name):
    try:
        name = name.replace('|', '')
        name = name.replace('?', '')
        name = name.replace(':', '')
        name = name.replace(';', '')
        name = name.replace("'", '')
        name = name.replace('"', '')
        name = name.replace(',', '')
        name = name.replace('.', '')
        name = name.replace('(', '')
        name = name.replace(')', '')
        name = name.replace('\n', '')
        name = name.replace('&', '')
        name = name.replace('।', '')
    except:
        pass
    tokens = tokenization(name)
    cleanedStop = clean_stopwords(tokens, extractStop())
    cleanedStop = ' '.join(i for i in cleanedStop)

    return lemmatization(cleanedStop)


def eachNews(df):
    tokenData = []
    for i in df.index:
        temp = Preprocesing(df['News'][i])
        tokenData.append([temp, df['Label'][i]])

    df = pd.DataFrame(tokenData, columns=['News', 'Label'])
    return df
