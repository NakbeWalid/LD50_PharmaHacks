"""
Feature pipeline aligned with ld50_starter.ipynb (Downloads / project notebook).

CELL 3: Morgan fingerprint (r=2, n_bits=1024).
CELL 5: Advanced descriptors — 10 physicochemical + 167 MACCS keys.
CELL 6: Merge + StandardScaler on the 10 continuous physicochemical columns only.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, rdFingerprintGenerator

MORGAN_RADIUS = 2
MORGAN_NBITS = 1024

# Column names for the advanced DataFrame (same order as the notebook).
COLS_PHYS = [
    "MolWt",
    "LogP",
    "HBD",
    "HBA",
    "TPSA",
    "Charge",
    "RotBonds",
    "Bertz",
    "AromRings",
    "fSP3",
]
ADV_COLS = COLS_PHYS + [f"MACCS_{i}" for i in range(167)]
COLS_TO_SCALE = COLS_PHYS.copy()

_morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
    radius=MORGAN_RADIUS, fpSize=MORGAN_NBITS
)


def smiles_to_morgan(smiles: str) -> np.ndarray:
    """Convert a SMILES string into a Morgan fingerprint (numpy binary vector)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(MORGAN_NBITS, dtype=np.int8)
    fp = _morgan_gen.GetFingerprint(mol)
    arr = np.zeros(MORGAN_NBITS, dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def _gasteiger_max_charge(mol: Chem.Mol) -> float:
    # Chemical reactivity (charges) — notebook: max Gasteiger charge after AllChem.ComputeGasteigerCharges
    try:
        AllChem.ComputeGasteigerCharges(mol)
        charges: list[float] = []
        for i in range(mol.GetNumAtoms()):
            try:
                charges.append(float(mol.GetAtomWithIdx(i).GetProp("_GasteigerCharge")))
            except (KeyError, ValueError):
                continue
        return max(charges) if charges else 0.0
    except Exception:
        return 0.0


def get_advanced_descriptors(smiles: str) -> list[float | int]:
    """
    Extract physicochemical properties and toxicophores (MACCS) from a molecule.
    We have 10 physicochemical descriptors + 167 MACCS keys (key 0 is always empty in RDKit).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [0.0] * 10 + [0] * 167

    # Molecular weight (g/mol)
    mw = Descriptors.MolWt(mol)

    # Predicted octanol-water partition coefficient (logP): lipophilicity measure
    logp = Descriptors.MolLogP(mol)

    # Number of hydrogen-bond donors (e.g., -OH, -NH groups)
    hbd = Descriptors.NumHDonors(mol)

    # Number of hydrogen-bond acceptors (e.g., O, N atoms with lone pairs)
    hba = Descriptors.NumHAcceptors(mol)

    # Topological Polar Surface Area: sum of polar surface contributions, correlates with permeability
    tpsa = Descriptors.TPSA(mol)

    charge = _gasteiger_max_charge(mol)

    # Flexibility (rotatable bonds)
    rot = Descriptors.NumRotatableBonds(mol)

    # Shape complexity (BertzCT)
    bertz = Descriptors.BertzCT(mol)

    # Number of aromatic rings (strong link with toxicity)
    arom = Descriptors.NumAromaticRings(mol)

    # Carbon richness (FractionCSP3) — indicates whether the molecule is "flat" or 3D
    fsp3 = Descriptors.FractionCSP3(mol)

    physchem_features = [mw, logp, hbd, hba, tpsa, charge, rot, bertz, arom, fsp3]

    maccs_fp = MACCSkeys.GenMACCSKeys(mol)
    maccs_arr = np.zeros(167, dtype=np.int8)
    DataStructs.ConvertToNumpyArray(maccs_fp, maccs_arr)

    return physchem_features + maccs_arr.tolist()
