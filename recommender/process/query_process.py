from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import  TruncatedSVD 
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import json , os,sys
import pickle
import spacy

from nltk.stem import WordNetLemmatizer

sys.path.append(".")

from .document_process import LemmaTokenizer


class QueryProcess():
    
    # load the model
    def load_model(self, model_path , vect_path):
        model = pickle.load( open (model_path, "rb"))
        vect = pickle.load( open (vect_path, "rb"))
        return model , vect

    # load the model
    def load_model(self , model_trained_path , vect_trained_path):
        model_trained = pickle.load( open (model_trained_path, "rb"))
        vect_trained = pickle.load( open (vect_trained_path, "rb"))
        return model_trained , vect_trained


    def test_similarity_examples(self, model , origin_space , vect , bow , query):

        bow_of_query = vect.transform( [query] )
        lsa = model.transform(bow_of_query)

        cosine = cosine_similarity(origin_space , lsa)

        similarity = np.argsort(cosine.reshape(-1))
        return cosine , similarity


    def run(self, query):
        
        SITE_ROOT = os.path.dirname(os.path.realpath(__file__))
        lsa_model , tfidf_vectorizer = self.load_model(SITE_ROOT + "/../models/lsa_sklearn.pkl" , SITE_ROOT + "/../models/vectorizer_sklearn.pkl")

        lsa , _ = self.load_model(SITE_ROOT + "/../models/model_space.pkl" , SITE_ROOT + "/../dictionary/bag_of_words.pkl")

        cos , sim = self.test_similarity_examples(lsa_model , lsa , tfidf_vectorizer , _ , query)

        paperId = []
        # load paperIds
        with open(SITE_ROOT + "/../models/paperId_dataset.json", "r") as f:
            paperId = json.load(f)

        count = 0
        paper_id = []
        for x in reversed(sim):
            if count == 12:
                break
            count += 1
            paper_id.append(paperId[x])


        return paper_id



if __name__ == "__main__":
    qp = QueryProcess()
    qp.run("Phenomena pertaining to galaxies or the Milky Way")
