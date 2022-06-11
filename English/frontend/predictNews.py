from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import eel
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras

eel.init('web')


def tokenization(data):
    tokenizer = RegexpTokenizer(r'\s+', gaps=True)
    tokenData = []
    tokenData.append(tokenizer.tokenize(data[0]))
    return tokenData


def clean_stopwords(tokendata):
    try:
        sw = stopwords.words('english')
    except:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('averaged_perceptron_tagger')
        sw = stopwords.words('english')
    clean_data = []
    for data in tokendata:
        clean_text = [words.lower()
                      for words in data if words.lower() not in sw]
        clean_data.append(clean_text)
    return clean_data


def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def lemmatize_sentence(sentence):
    lemmatizer = WordNetLemmatizer()
    nltk_tagged = nltk.pos_tag(sentence)
    wordnet_tagged = map(lambda x: (
        x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            lemmatized_sentence.append(word)
        else:
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)


@eel.expose
def predict(news):
    trainR = pd.read_csv('Lemma_Sent.csv')
    trainR = trainR['text'].tolist()

    tokenData = tokenization([news])
    clean_data = clean_stopwords(tokenData)
    r = []
    clean_data = clean_stopwords(tokenData)
    for i in range(len(clean_data)):
        temp = lemmatize_sentence(clean_data[i])
        r.append(temp)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(trainR)
    check = tokenizer.texts_to_sequences(r)

    check_trunc = pad_sequences(check, maxlen=1000)

    check_trunc = pad_sequences(check, maxlen=1000)

    cnnModel = keras.models.load_model('model.h5')
    pred = cnnModel.predict(check_trunc) > 0.5

    return "The given news is: " + str(pred[0][0])


eel.start('index.html', size=(1000, 600))
