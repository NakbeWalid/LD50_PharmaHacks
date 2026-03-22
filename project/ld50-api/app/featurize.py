"""Same feature pipeline as ld50_starter.ipynb: Morgan (r=2, 1024) + 5 phys-chem + MACCS (167)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator

MORGAN_RADIUS = 2
MORGAN_NBITS = 1024

ADV_COLS = ["MolWt", "LogP", "HBD", "HBA", "TPSA"] + [f"MACCS_{i}" for i in range(167)]
COLS_TO_SCALE = ["MolWt", "LogP", "HBD", "HBA", "TPSA"]

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


def get_advanced_descriptors(smiles: str) -> list[float | int]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [0.0] * 5 + [0] * 167
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)
    physchem = [mw, logp, hbd, hba, tpsa]
    maccs_fp = MACCSkeys.GenMACCSKeys(mol)
    maccs_arr = np.zeros(167, dtype=np.int8)
    DataStructs.ConvertToNumpyArray(maccs_fp, maccs_arr)
    return physchem + maccs_arr.tolist()


def smiles_to_feature_frame(smiles: str) -> pd.DataFrame:
    morgan = smiles_to_morgan(smiles)
    adv = get_advanced_descriptors(smiles)
    morgan_df = pd.DataFrame([morgan], columns=list(range(MORGAN_NBITS)))
    adv_df = pd.DataFrame([adv], columns=ADV_COLS)
    return pd.concat([morgan_df, adv_df], axis=1)
