import os
import pickle

from datasets import load_dataset
import fasttext
import fasttext.util
import numpy as np
import openai
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, BertModel, BertTokenizer, DistilBertModel, RobertaTokenizer, RobertaModel

import vsm

DATA_HOME = os.path.join('data', 'wordrelatedness')
VSM_HOME = os.path.join('data', 'vsmdata')

# Levers
USE_LSA = False
USE_PPMI = False
USE_TRANSFORMER = False

TEST_SIZE = 0.2

os.chdir("HWS/HW1")

dev_df = pd.read_csv(os.path.join(DATA_HOME, "cs224u-wordrelatedness-dev.csv"))
count_df = pd.read_csv(os.path.join(VSM_HOME, "giga_window5-scaled.csv.gz"), index_col=0)

count_pred_df, count_rho = vsm.word_relatedness_evaluation(dev_df, count_df)

cur_df = count_df

task_index = pd.read_csv(
os.path.join(VSM_HOME, 'yelp_window5-scaled.csv.gz'),
usecols=[0], index_col=0)

full_task_vocab = list(task_index.index)
# train_dev_df = dev_df.sample(frac=TEST_SIZE)
# test_dev_df = dev_df.drop(index=train_dev_df.index)

# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=test_size
#     )

if USE_PPMI:
    
    print("Applying PPMI...")
    cur_df = vsm.pmi(count_df, positive=True)
    print("PPMI complete")
    
    pred_df, rho = vsm.word_relatedness_evaluation(dev_df, cur_df)
    print("Only PPMI rho", rho)
    
    if USE_LSA:
        pmi_k = 100
        
        print("Applying LSA to PPMI...")
        cur_df = vsm.lsa(cur_df, k=pmi_k)
        print("LSA complete")
        pred_df, rho = vsm.word_relatedness_evaluation(dev_df, cur_df)
        
        print("PPMI and LSA rho", rho)
    
if USE_LSA:
    pmi_k = 100
    
    print("Applying LSA alone...")
    cur_df = vsm.lsa(count_df, k=pmi_k)
    print("LSA complete")
    pred_df, rho = vsm.word_relatedness_evaluation(dev_df, cur_df)
    
    print("Only LSA rho", rho)

if USE_TRANSFORMER:
    
    # Hyperparameters for transformer
    layer = 1
    pool_func = vsm.max_pooling
    bert_weights_name = 'bert-base-uncased'
    
    tokenizer = BertTokenizer.from_pretrained(bert_weights_name)
    model = BertModel.from_pretrained(bert_weights_name)
    vocab = list(set(dev_df.word1.values) | set(dev_df.word2.values))
    
    print("Pooling subwords...")
    cur_df = vsm.create_subword_pooling_vsm(
        vocab,
        tokenizer,
        model,
        layer=layer,
        pool_func=pool_func
        )
    print("Pooling complete")
    
    pred_df, rho = vsm.word_relatedness_evaluation(dev_df, cur_df)
    
    print("Only BERT subword rho", rho)
    
    if USE_LSA:
        
        pmi_k = 300
        
        print("Applying LSA to BERT...")
        cur_df = vsm.lsa(count_df, k=pmi_k)
        print("LSA complete")
        pred_df, rho = vsm.word_relatedness_evaluation(dev_df, cur_df)
        
        print(rho)
        
def baseline_grid_search_transformer(dev_df):
    layers = [1]
    pool_funcs = [vsm.mean_pooling]
    bert_weights = ['distilbert-base-uncased']
    
    vocab = list(set(dev_df.word1.values) | set(dev_df.word2.values))
    
    for bert_weights_name in bert_weights:
        tokenizer = AutoTokenizer.from_pretrained(bert_weights_name)
        model = DistilBertModel.from_pretrained(bert_weights_name)
        for pool_func in pool_funcs:
            for layer in layers:
                print("Pooling subwords...")
                print(f"Parameters model: {bert_weights_name}, pooling function: {pool_func}, layer: {layer}")
                cur_df = vsm.create_subword_pooling_vsm(
                    vocab,
                    tokenizer,
                    model,
                    layer=layer,
                    pool_func=pool_func
                    )
                print("Pooling complete")
                pred_df, rho = vsm.word_relatedness_evaluation(dev_df, cur_df)
                print("Score is", rho, "\n")
                
def tune_LSA(dev_df, count_df):
    ks = [25, 50, 75, 100, 125, 150, 175, 200]
    
    for k in ks:
        cur_df = vsm.lsa(count_df, k=k)
        pred_df, rho = vsm.word_relatedness_evaluation(dev_df, cur_df)
        print(f"LSA score with k={k} is", rho)
        
def ft_scores(dev_df):
    fasttext.util.download_model('en', if_exists='ignore')
    ft = fasttext.load_model('cc.en.300.bin')
    fasttext.util.reduce_model(ft, 200)
    print(ft.get_dimension())
    vocab = list(set(dev_df.word1.values) | set(dev_df.word2.values))
    vectors = {}
    print("Getting fasttext vectors...")
    for word in vocab:
        vectors[word] = ft.get_word_vector(word)
    print("Converting dictionary to pandas df...")
    vsm_df = pd.DataFrame.from_dict(vectors, orient='index')
    pred_df, rho = vsm.word_relatedness_evaluation(dev_df, vsm_df)
    print("Score is", rho, "\n")


def bert_to_static(dev_df):
    dataset = load_dataset("bookcorpus")
    


if __name__ == "__main__":
    # baseline_grid_search_transformer(dev_df)
    tune_LSA(dev_df, count_df)
    
    # bert_weights_name = 'bert-base-uncased'
    # tokenizer = BertTokenizer.from_pretrained(bert_weights_name)
    # sentence = 'The company abandoned its efforts to enter the foreign market after encountering numerous obstacles.'
    # print(tokenizer._tokenize(sentence))
    # ft_scores(dev_df)
    
    # from transformers import pipeline
    # generator = pipeline("text2text-generation", model="google/flan-t5-xl")
    # print(generator("Give a sentences using the word 'abandon' in its current form"))
    # from transformers import T5Tokenizer, T5ForConditionalGeneration

    # tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
    # model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")

    # input_text = "translate English to German: How old are you?"
    # input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    # outputs = model.generate(input_ids)
    # print(tokenizer.decode(outputs[0]))
    
    # dataset = load_dataset("bookcorpus")
    # filtered_text = []
    # for i in range(len(dataset['train'])):
    #     text = dataset['train'][i]['text']
    #     words = text.split()
    #     if len(words) >= 7:
    #         filtered_text.append(text)
    #     if i % 100000 == 0:
    #         print(i, "texts filtered")
            
    # with open('filtered_bookcorpus_list', 'wb') as fp:
    #     pickle.dump(filtered_text, fp)
    #     print('Done writing list into a binary file')
    
    # with open("filtered_bookcorpus_list", 'rb') as f:
    #     texts = pickle.load(f)

    # texts_split = []
    # for i in range(len(texts)):
    #     words = set(texts[i].split())
    #     texts_split.append(words)
    #     if i % 100000 == 0:
    #         print(i, "texts added")
    
    # with open('texts_split', 'wb') as fp:
    #     pickle.dump(texts_split, fp)
    #     print('Done writing list into a binary file')
        
        
    
    


    
        

