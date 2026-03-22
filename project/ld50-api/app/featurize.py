"""Morgan fingerprint + RDKit physicochemical/MACCS features for LD50 models."""

from __future__ import annotations

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, rdFingerprintGenerator

MORGAN_RADIUS = 2
MORGAN_NBITS = 1024

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
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(MORGAN_NBITS, dtype=np.int8)
    fp = _morgan_gen.GetFingerprint(mol)
    arr = np.zeros(MORGAN_NBITS, dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def _gasteiger_max_charge(mol: Chem.Mol) -> float:
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
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [0.0] * 10 + [0] * 167

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)
    charge = _gasteiger_max_charge(mol)
    rot = Descriptors.NumRotatableBonds(mol)
    bertz = Descriptors.BertzCT(mol)
    arom = Descriptors.NumAromaticRings(mol)
    fsp3 = Descriptors.FractionCSP3(mol)

    physchem_features = [mw, logp, hbd, hba, tpsa, charge, rot, bertz, arom, fsp3]

    maccs_fp = MACCSkeys.GenMACCSKeys(mol)
    maccs_arr = np.zeros(167, dtype=np.int8)
    DataStructs.ConvertToNumpyArray(maccs_fp, maccs_arr)

    return physchem_features + maccs_arr.tolist()
