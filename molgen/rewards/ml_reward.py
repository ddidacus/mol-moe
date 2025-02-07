from mol2vec.features import mol2alt_sentence, MolSentence
from gensim.models import Word2Vec
from rdkit import Chem
import numpy as np
import joblib

class ClassificationReward:

    def __init__(self, model, normalize=False, mean=0, std=1, maximize=True):
        """
        Args:
            model:str       path to the trained ML model
            mean:float      center of data distribution for normalized outputs
            std:float       st. dev of =
            normalize:bool  gaussian normalize data with mean,std
            maximize:bool   flip predictions if the RL objective is minimization
        """
        self.m2vmodel = Word2Vec.load("support/mol2vec_model_300dim.pkl")
        self.model = joblib.load(model)

    def sentences2vec(self, sentences: list, model, unseen=None) -> np.ndarray:
        """
        Convert SMILES to numpy array with Mol2Vec
        Args:
            sentences:list  list of SMILES
            model:object    pickled mol2vec
        Returns:
            y:np.array      encoded dense mol vector
        """
        keys = set(model.wv.key_to_index.keys())
        vec = []
        if unseen:
            unseen_vec = model.wv.get_vector(unseen)

        for sentence in sentences:
            if unseen:
                vec.append(
                    sum([
                        model.wv.get_vector(y) if y in set(sentence) &
                        keys else unseen_vec for y in sentence
                    ]))
            else:
                vec.append(
                    sum([
                        model.wv.get_vector(y)
                        for y in sentence
                        if y in set(sentence) & keys
                    ]))
        return np.array(vec)

    def __call__(self, x):
        """
        Parse SMILES and do regression with provided model
        Args:
            x:str       molecule in SMILES format
        Returns:
            y:float     predicted ADME activity
        """
        try:
            mol = Chem.MolFromSmiles(str(x))
            sentence = MolSentence(mol2alt_sentence(mol, 1))
            encoding = self.sentences2vec([sentence], self.m2vmodel, unseen='UNK')[0]
            logits = self.model.predict_proba([encoding])[0]
            return logits[1]
        except: return 0