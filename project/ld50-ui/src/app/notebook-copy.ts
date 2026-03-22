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

  pipelineTitle: 'Our pipeline',
  pipelineBullets: [
    'Random split 70% / 10% / 20% (TDC-style).',
    'Morgan fingerprint (radius 2, 1024 bits).',
    'Ten physicochemical descriptors (MolWt, LogP, HBD, HBA, TPSA, Charge, RotBonds, Bertz, AromRings, fSP3) plus 167 MACCS keys; StandardScaler on the 10 continuous columns only.',
    'XGBoost with fixed hyperparameters; benchmark vs Random Forest and SVR on the validation set; SHAP global summary and per-molecule waterfall on the validation set.',
  ],

  chartsTitle: 'Figures',
  chartsLead:
    'Validation scatter (predictions vs ground truth). SHAP summary (top 10 features). SHAP waterfall for one validation molecule.',

  metricsLead: `Each row is one data split. All metrics are computed on the target log(LD50). Mean absolute error (MAE) and root mean squared error (RMSE) use the same units as the target; lower is better. R² is the coefficient of determination (share of variance explained); higher is better, up to 1.`,
} as const;
