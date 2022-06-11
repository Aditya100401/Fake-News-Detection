import eel
import preProcessing
import pandas as pd
from transformers import AutoModel, AutoTokenizer
import torch
import keras
import numpy as np
import dill


path_muril = 'google/muril-base-cased'
tokenizer_muril = AutoTokenizer.from_pretrained(path_muril)
model_muril = AutoModel.from_pretrained(path_muril, output_hidden_states=True)

path_indic = 'ai4bharat/indic-bert'
tokenizer_indic = AutoTokenizer.from_pretrained(path_indic)
model_indic = AutoModel.from_pretrained(path_indic, output_hidden_states=True)

cnnMuril = keras.models.load_model('muril_cnn.h5')
cnnIndic = keras.models.load_model('indic_cnn.h5')

svmMuril = dill.load(open('muril_svm.sav', 'rb'))
svmIndic = dill.load(open('indic_svm.sav', 'rb'))

logMuril = dill.load(open('muril_log.sav', 'rb'))
logIndic = dill.load(open('indic_log.sav', 'rb'))

eel.init('web')


def muril_embed(news):
    temp = tokenizer_muril.convert_ids_to_tokens(tokenizer_muril.encode(news))
    input_encoded = tokenizer_muril.encode_plus(news, return_tensors="pt")

    with torch.no_grad():
        states = model_muril(**input_encoded).hidden_states
    output = torch.stack([states[i] for i in range(len(states))])

    token_vecs = output[-2][0]
    sentence_embedding = torch.mean(token_vecs, dim=0)
    sentence_embedding = sentence_embedding.tolist()
    sentence_embedding = np.array(sentence_embedding).reshape(1, -1)
    return sentence_embedding


def indic_embed(news):
    temp = tokenizer_indic.convert_ids_to_tokens(tokenizer_indic.encode(news))
    input_encoded = tokenizer_indic.encode_plus(news, return_tensors="pt")

    with torch.no_grad():
        states = model_indic(**input_encoded).hidden_states
    output = torch.stack([states[i] for i in range(len(states))])

    token_vecs = output[-2][0]
    sentence_embedding = torch.mean(token_vecs, dim=0)
    sentence_embedding = sentence_embedding.tolist()
    sentence_embedding = np.array(sentence_embedding).reshape(1, -1)
    return sentence_embedding


def cnn_muril(sentence_embedding):
    pred = cnnMuril.predict(sentence_embedding) > 0.5
    return "The given news is: " + str(pred[0][0])


def cnn_indic(sentence_embedding):
    pred = cnnIndic.predict(sentence_embedding) > 0.5
    return "The given news is: " + str(pred[0][0])


def svm_muril(sentence_embeding):
    pred = svmMuril.predict(sentence_embeding)
    if pred[0] == 0:
        return "The given news is: False"
    else:
        return "The given news is: True"


def svm_indic(sentence_embedding):
    pred = svmIndic.predict(sentence_embedding)
    if pred[0] == 0:
        return "The given news is: False"
    else:
        return "The given news is: True"


def log_muril(sentence_embedding):
    pred = logMuril.predict(sentence_embedding)
    if pred[0] == 0:
        return "The given news is: False"
    else:
        return "The given news is: True"


def log_indic(sentence_embedding):
    pred = logIndic.predict(sentence_embedding)
    if pred[0] == 0:
        return "The given news is: False"
    else:
        return "The given news is: True"


@eel.expose
def predict(news, model, embedding):
    news = preProcessing.Preprocesing(news)

    if embedding == 'MuRIL':
        sent_embed = muril_embed(news)
    elif embedding == 'IndicBERT':
        sent_embed = indic_embed(news)

    if model == 'CNN' and embedding == 'MuRIL':
        return cnn_muril(sent_embed)

    if model == 'CNN' and embedding == 'IndicBERT':
        return cnn_indic(sent_embed)

    if model == 'SVM' and embedding == 'MuRIL':
        return svm_muril(sent_embed)

    if model == 'SVM' and embedding == 'IndicBERT':
        return svm_indic(sent_embed)

    if model == 'Logistic' and embedding == 'MuRIL':
        return log_muril(sent_embed)

    if model == 'Logistic' and embedding == 'IndicBERT':
        return log_indic(sent_embed)


eel.start('index.html', size=(1000, 600))
