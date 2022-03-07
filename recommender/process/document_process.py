from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import  TruncatedSVD 


import numpy as np
import json
import re


import spacy


from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

import nltk
from nltk.corpus import stopwords

import pickle

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stop_words = set(stopwords.words('english')) 



## either upload from json file or directly from mongoDB



# Interface lemma tokenizer from nltk with sklearn
class LemmaTokenizer:
    ignore_tokens = [',', '.', ';', ':', '"', '``', "''", '`']
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        #self.stm = PorterStemmer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if t not in self.ignore_tokens]

class DocumentProcess():

    def __init__(self):
        self.datafile = "../new_arxiv_no_null.json" ## it's better to use data from database directly
        self.run()

    def load_data(self,datafile):
    
        dataset = []
        with open(datafile , 'r') as f:
            dataset = json.load(f)
        return dataset




    ## remove unnecessary characters
    def cleanText(self, documents):
        """
            this function take documents (text) as input and cleaning the document from unnecassary characters and whitespaces . then return the document
            
        """

        CLEANR = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
        paperId_dataset = []
        abstract_dataset = []

        for x in documents:
            cleantext = re.sub(CLEANR, ' ', str(x["abstract"]))
            cleantext = re.sub(r"[^a-zA-Z ]"," ",cleantext)
            cleantext = re.sub("arXiv","",cleantext)
            cleantext = re.sub(" +"," ",cleantext)
            abstract_dataset.append(cleantext)
            paperId_dataset.append( x["paperId"] )


        return paperId_dataset , abstract_dataset 



    def lemmatization(self, texts, allowed_postags=["PROPN","NOUN", "ADJ", "VERB", "ADV"]):

        #def lemmatization(texts, allowed_postags=["PROPN","NOUN", "ADJ", "ADV"]):
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        texts_out = []
        for text in texts:
            doc = nlp(text)
            new_text = []
            for token in doc:
                if token.pos_ in allowed_postags:
                    if len(token.lemma_) > 2:
                        new_text.append(token.lemma_)
            
            final = " ".join(new_text)
            texts_out.append(final)
        return (texts_out)


    def lsa(self, dataset, topic_n=50 , max_features=50000):

        if dataset is None:
            return

        # Lemmatize the stop words
        tokenizer=LemmaTokenizer()
        token_stop = tokenizer(' '.join(stop_words))


        lemma_dataset = self.lemmatization(dataset)

        tfidf_vectorizer = TfidfVectorizer( max_features=max_features,  stop_words=token_stop, max_df=0.5 ,use_idf=True, ngram_range=(1,1), tokenizer=tokenizer)

        #sparse matrix of (n_samples, n_features)
        #Tf-idf-weighted document-term matrix.
        bag_of_words = tfidf_vectorizer.fit_transform( lemma_dataset)



        lsa_model = TruncatedSVD(n_components=topic_n, algorithm='arpack', n_iter=50, random_state=122)

        # Reduced version of X. This will always be a dense array.
        # ndarray of shape (n_samples, n_components)
        lsa = lsa_model.fit_transform(bag_of_words)

        return lsa_model , lsa , tfidf_vectorizer , bag_of_words



    # save the model
    def save_model(self , model, model_path, vect, vect_path):
        pickle.dump(model, open(model_path, 'wb'))
        pickle.dump(vect, open(vect_path, 'wb'))
        print("Save model and vect into {model_path} {vect_path}")

    def save_model(self , model_trained, model_trained_path , vect_trained, vect_trained_path):
        pickle.dump(model_trained, open(model_trained_path, 'wb'))
        pickle.dump(vect_trained, open( vect_trained_path , 'wb'))
        print("Save model sapce and bag_of_words into {model_path} {vect_path}")

    def run(self ):
        dataset = self.load_data(self.datafile)
        paperId_dataset , abstract_dataset = self.cleanText(dataset)

        # save paperIds
        with open("../models/paperId_dataset.json", "w") as f:
            f.write(json.dumps(paperId_dataset, sort_keys=False, indent=4, separators=(',', ': ')))
            print("Save paper_id to json")



        lsa_model , lsa , tfidf_vectorizer , bag_of_words = self.lsa(abstract_dataset)

        self.save_model( lsa_model , "../models/lsa_sklearn.pkl" , tfidf_vectorizer , "../models/vectorizer_sklearn.pkl" )
        self.save_model( lsa , "../models/model_space.pkl" , bag_of_words , "../dictionary/bag_of_words.pkl" )
        

