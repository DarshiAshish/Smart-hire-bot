import numpy as np
import pandas as pd
import torch
import transformers as ppb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
import time

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




def main():
    df = read_data("hrdata.csv")
    df.sort_values(by='Intent', inplace=True)
    df.to_csv("checking.csv")
    df = preprocessing(df)
    model_list = ["distil_bert","roberta"]

    for each in model_list:
        print("-------------------------",each,"------------------------------------------", flush=True)
        start_time = time.time()
        if each == "distil_bert":
            model, tokenizer = distil_bert_model()
            features, labels, max_len = intermediate_preprocessing_steps(df, tokenizer, model)
            np.savez("features_distilbert.npz",features)
        if each == "roberta":
            model, tokenizer = roberta_model()
            features, labels, max_len = intermediate_preprocessing_steps(df, tokenizer, model)
            np.savez("features_roberta.npz",features)


        if each == "distil_bert":
            max_len_path = "max_len/distil_bert/length.txt"
        if each == "roberta":
            max_len_path = "max_len/roberta/length.txt"
        if each in ["distil-bert","roberta"]:
            with open(max_len_path,"w") as f:
                f.write(str(max_len))

        print("features and labels are generated for {each}", flush=True)
        print("------------------------------------------------", flush=True)
        embeddings_end_time = time.time()
        
        

main()
