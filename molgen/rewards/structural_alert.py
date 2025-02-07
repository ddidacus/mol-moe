from rdkit import rdBase
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import numpy as np

class StructuralAlertReward:
    def __init__(self, file):
        self.structures = []
        with open(file, 'r') as f:
            self.structures = [line.strip() for line in f]

    def has_substructure(self, smiles, smarts):
        try:
            mol = Chem.MolFromSmiles(smiles)
            struct = Chem.MolFromSmarts(smarts)
            return mol.HasSubstructMatch(struct)
        except: return True # default: failed processing is penalized

    def __call__(self, smiles):
        """ Reward is the fraction of structures not present """
        # If I computed the mean, the penalty would be squeezed with an increasing number of structures
        is_clear = not np.all([int(self.has_substructure(smiles=smiles, smarts=struct)) for struct in self.structures])
        return float(is_clear)

    
