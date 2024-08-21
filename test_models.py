import numpy as np
import pandas as pd
import faiss
import torch
import transformers as ppb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
import time
import pickle
import warnings
warnings.filterwarnings("ignore")

# reading data
def read_data(path):
    df = pd.read_csv(path)
    return df


# remove stop words
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
S=set(stopwords.words('english'))
def remove_stop(text):
    p=" ".join([i for i in text.split() if i not in S])
    return p

# Removing punctuation
import string
def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree


# Label encoder
def preprocessing(df):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    getL = le.fit_transform(df['Intent'])
    df['intent'] = getL
    # saving the labels
    df.to_csv('labels_generated.csv')
    return df



def intermediate_preprocessing_steps(df, tokenizer, model):
    tokenized = df['Text'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
    max_len = 0
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)

    padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
    attention_mask = np.where(padded != 0, 1, 0)
    attention_mask.shape
    input_ids = torch.tensor(padded)  
    attention_mask = torch.tensor(attention_mask)
    with torch.no_grad():
        last_hidden_states = model(input_ids,attention_mask)
    features = last_hidden_states[0][:,0,:].numpy()
    labels = df['intent']
    return features, labels, max_len



def distil_bert_model():
    model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)
    return model,tokenizer


def roberta_model():
    from transformers import RobertaTokenizer, RobertaModel
    import torch
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaModel.from_pretrained("roberta-base")
    return model, tokenizer


def out_labels(labels_path):
    ldf = pd.read_csv(labels_path)
    edf = ldf[['Intent', 'intent']].drop_duplicates()
    json_res = {}
    for each, each_intent in zip(edf["intent"], edf["Intent"]):
        json_res[int(each)] = each_intent
    return json_res


out_labels_json = out_labels("labels_generated.csv")
# print(get_category(query, out_labels_json))

model1, tokenizer1 = distil_bert_model()
model2, tokenizer2 = roberta_model()
max_len = 28
random_model = pickle.load(open('distil_bert/random.sav', 'rb'))
def test_samples(model, tokenizer, text, max_len):
    
    testing = [text]
    testing= pd.Series(testing)
    testing = testing.apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
    # testing_encode = tokenizer.encode(testing,add_special_tokens=True)
    test_pad = np.array([i + [0]*(max_len-len(i)) for i in testing.values])
    attention_mask = np.where(test_pad != 0, 1, 0)
    input_ids = torch.tensor(test_pad)  
    attention_mask = torch.tensor(attention_mask)

    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)
    checking = last_hidden_states[0][:,0,:].numpy()
    return checking

def ml_model(ml_model, encodings):
    out = ml_model.predict(encodings)
    return out_labels_json[out[0]]

def main():
    text = "When does earned leaves fall in balance"
    print(ml_model(random_model,test_samples(model1, tokenizer1, text, max_len)))

main()