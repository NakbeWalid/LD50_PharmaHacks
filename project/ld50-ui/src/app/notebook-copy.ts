/** Static English copy aligned with ld50_starter.ipynb (project / starter notebook). */

export const NB = {
  title: 'LD50 Toxicity Prediction',
  subtitle: 'PharmaHacks 2026 ML Challenge',

  introLead:
    'Predict the acute toxicity (LD50) of drug-like molecules from their chemical structure.',

  introBody: `The target (Y) is the log-transformed LD50 in mg/kg. Higher values mean less toxic; lower values mean more toxic. This is a regression problem. Each molecule is given as a SMILES string — a text encoding of molecular structure (e.g. CC(=O)Oc1ccccc1C(=O)O is aspirin).`,

  datasetTitle: 'The dataset',
  datasetBody: `We use the LD50_Zhu dataset from Therapeutics Data Commons (TDC), originally published in Zhu et al., Quantitative Structure-Activity Relationship Modeling of Rat Acute Toxicity by Oral Exposure, Chemical Research in Toxicology (2009).

LD50 (Lethal Dose, 50%) is the dose required to kill 50% of a test population (rats, oral). It is a fundamental measure of acute toxicity.

The dataset contains about 7,400 molecules. Each row has Drug_ID, Drug (SMILES), and Y (log LD50 in mg/kg). Task: predict the continuous log(LD50) from structure.`,

  roadmapTitle: 'Suggested roadmap',
  roadmapSteps: [
    'Explore the data: plot the target distribution, summary statistics, outliers, skew.',
    'Featurize molecules: SMILES must become numbers (fingerprints, descriptors, MACCS keys, etc.).',
    'Train a baseline (e.g. Ridge or Random Forest) and evaluate with R², MAE, RMSE on validation.',
    'Iterate: stronger models (XGBoost), tuning, better features, error analysis.',
  ],

  pipelineTitle: 'Our pipeline (same cells as the notebook)',
  pipelineBullets: [
    'Random split 70% / 10% / 20% (TDC-style).',
    'CELL 3: Morgan fingerprint (radius 2, 1024 bits).',
    'CELL 5–6: 10 physicochemical descriptors (MolWt, LogP, HBD, HBA, TPSA, Charge, RotBonds, Bertz, AromRings, fSP3) + 167 MACCS keys; StandardScaler on the 10 continuous columns only.',
    'CELL 7: XGBoost with notebook hyperparameters; CELL 7.5: benchmark vs Random Forest & SVR on validation; CELL 8–9: SHAP on the validation set.',
  ],

  chartsTitle: 'Figures (same as notebook plots)',
  chartsLead:
    'CELL 7: validation scatter (predictions vs ground truth). CELL 8: SHAP summary (top 10). CELL 9: SHAP waterfall for one validation molecule.',
} as const;
