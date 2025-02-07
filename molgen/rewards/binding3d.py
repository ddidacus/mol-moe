import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, rdShapeHelpers
from rdkit.Chem import rdShapeAlign
import rdmolutils

class BindingSimilarity3DReward:
    def __init__(self):
        sdf_file = "5IDN_ligand_xray.sdf"
        supplier = Chem.SDMolSupplier(sdf_file)
        self.ref_mol = supplier[0]  # Assuming the first molecule is the reference

    def __call__(self, x) -> float:
        # try:
        reward = calculate_3d_similarity(x, self.ref_mol)
        return reward
        # except:
        #     return float(0.0)


def calculate_3d_similarity(smiles, ref_mol, num_confs=100):
    shape_similarities = []
    mol = generate_conformers_fromSMILES(smiles, num_confs=num_confs)
    for conf in mol.GetConformers():
        shape_similarity, color_similarity = rdShapeAlign.AlignMol(ref_mol, mol, probeConfId=conf.GetId())
        shape_similarities.append(shape_similarity)
    return max(shape_similarities)


def generate_conformers_fromSMILES(smiles: str, num_confs: int = 100):
    """Generate conformers for a given SMILES string."""
    # Convert SMILES to RDKit molecule
    mol = Chem.MolFromSmiles(smiles)
    mol = rdmolutils.embed_mol(mol)
    mol.SetProp("_Name", "0")
    mol = rdmolutils.generate_confs(mol, num_confs=num_confs)
    return mol