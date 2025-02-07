from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
import math
import numpy as np

class CNSMPOReward:
    def __init__(self, aggregation="sum"):
        m_clogP = lambda x: map_logits(x, 3, 5, False)
        m_clogD = lambda x: map_logits(x, 2, 4, False)
        m_MW = lambda x: map_logits(x, 360, 500, False)
        m_TPSA = lambda x: map_functions(x, [lambda x: map_logits(x, 20, 40, True), lambda x: map_logits(x, 90, 120, False)], [60, 200])(x)
        m_HBD = lambda x: map_logits(x, 0, 4, False)
        # https://pubs.acs.org/doi/10.1021/acschemneuro.6b00029
        self.drug_likeness_descriptors = lambda x: [Crippen.MolLogP(x), calculate_clogd(x), Descriptors.ExactMolWt(x), Descriptors.TPSA(x), Descriptors.NumHDonors(x)]
        agg_fn = np.mean if aggregation == "mean" else np.sum
        self.CNSMPO = lambda x: agg_fn([m_clogP(x[0]), m_clogD(x[1]), m_MW(x[2]), m_TPSA(x[3]), m_HBD(x[4])])

    def __call__(self, x) -> float:
        try:
            mol = Chem.MolFromSmiles(str(x))
            features = self.drug_likeness_descriptors(mol)
            return float(self.CNSMPO(features))
        except: return float(0.0)

class clogPReward:
    def __init__(self):
        self.m_clogP = lambda x: map_logits(x, 3, 5, False)
    def __call__(self, x) -> float:
        try:
            mol = Chem.MolFromSmiles(str(x))
            return float(self.m_clogP(Crippen.MolLogP(mol)))
        except: return float(0.0)

class clogDReward:
    def __init__(self):
        self.m_clogD = lambda x: map_logits(x, 2, 4, False)
    def __call__(self, x) -> float:
        try:
            mol = Chem.MolFromSmiles(str(x))
            return float(self.m_clogD(calculate_clogd(mol)))
        except: return float(0.0)

class MWReward:
    def __init__(self):
        self.m_MW = lambda x: map_logits(x, 360, 500, False)
    def __call__(self, x) -> float:
        try:
            mol = Chem.MolFromSmiles(str(x))
            return float(self.m_MW(Descriptors.ExactMolWt(mol)))
        except: return float(0.0)

class TPSAReward:
    def __init__(self):
        self.m_TPSA = lambda x: map_functions(x, [lambda x: map_logits(x, 20, 40, True), lambda x: map_logits(x, 90, 120, False)], [60, 200])
    def __call__(self, x) -> float:
        try:
            mol = Chem.MolFromSmiles(str(x))
            return float(self.m_TPSA(Descriptors.TPSA(mol)))
        except: return float(0.0)

class HBDReward:
    def __init__(self):
        self.m_HBD = lambda x: map_logits(x, 0, 4, False)
    def __call__(self, x) -> float:
        try:
            mol = Chem.MolFromSmiles(str(x))
            return float(self.m_HBD(Descriptors.NumHDonors(mol)))
        except: return float(0.0)


def estimate_pka(mol):
    """ Approximation of the pKa value """
    return rdMolDescriptors.CalcNumLipinskiHBA(mol)

def calculate_clogd(mol, pH=7.4):
    """ Distribution coefficient at pH 7.4 """
    logp = Crippen.MolLogP(mol)
    pka = estimate_pka(mol)
    
    fraction_ionized = 1 / (1 + 10**(pka - pH))
    clogd = logp - math.log10(1 + 10**(pH - pka))
    
    return clogd

def map_logits(x, x_min, x_max, grows=True):
    """ Maps each function range to a normalized function """
    if grows:
        return (max(min(x, x_max), x_min) - x_min) / (x_max - x_min)
    else:
        return 1 - ((max(min(x, x_max), x_min) - x_min) / (x_max - x_min))

def map_functions(x, functions, ranges):
    """ Aggregate multiple normalized functions by ranges """
    ranges, functions = zip(*sorted(zip(ranges, functions)))
    fn = functions[-1]
    for i, t in enumerate(ranges):
        if x < t:
            fn = functions[i]
            break
    return fn
