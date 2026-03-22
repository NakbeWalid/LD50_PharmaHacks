export const PARAM_ORDER = [
  'n_estimators',
  'learning_rate',
  'max_depth',
  'subsample',
  'colsample_bytree',
  'reg_alpha',
  'reg_lambda',
  'early_stopping_rounds',
  'tree_method',
] as const;

export const PARAM_HINTS: Record<string, string> = {
  n_estimators:
    'Upper bound on boosting rounds; extra trees are unused if early stopping triggers first.',
  learning_rate:
    'Step size for each tree’s update — smaller values often need more trees but can generalize better.',
  max_depth:
    'How deep each tree can grow; deeper trees capture more interactions but may overfit.',
  subsample:
    'Fraction of training rows sampled per tree; values below 1 add randomness and can reduce overfitting.',
  colsample_bytree:
    'Fraction of input features sampled per tree; helps on high‑dimensional fingerprints.',
  reg_alpha:
    'L1 penalty on leaf weights — encourages sparser, simpler leaf values.',
  reg_lambda:
    'L2 penalty on leaf weights — smooths predictions and curbs large leaf scores.',
  early_stopping_rounds:
    'Stop if validation loss does not improve for this many consecutive rounds.',
  tree_method:
    'Tree construction algorithm — histogram is fast and scales well to large matrices.',
};
