export type SplitMetrics = {
  n: number;
  mae: number;
  rmse: number;
  r2: number;
};

export type MetricsBlock = {
  description?: string;
  split_rule?: string;
  train: SplitMetrics;
  valid: SplitMetrics;
  test: SplitMetrics;
};

export type BenchmarkRow = {
  model: string;
  r2_validation: number;
  mae_validation: number;
  train_time_s: number;
  train_rows_used?: number;
};

export type BenchmarkBlock = {
  description: string;
  r2_threshold: number;
  svr_note: string;
  leaderboard: BenchmarkRow[];
  rank_by_model: Record<string, number>;
};

export type Report = {
  metrics: MetricsBlock;
  plots: {
    scatter_validation: { y_true: number[]; y_pred: number[] };
    shap: {
      global_top10: { feature: string; mean_abs_shap: number }[];
      waterfall: {
        molecule_index: number;
        molecule_id: string;
        expected_value: number;
        predicted_log_ld50: number;
        actual_log_ld50: number;
        features: { feature: string; shap: number; value: number }[];
      };
    } | null;
  };
  dataset: {
    split_sizes: { train: number; valid: number; test: number };
    sample_rows: { Drug_ID: string; Drug: string; Y: number }[];
  };
  model: {
    name: string;
    params: Record<string, string | number>;
  };
  notebook?: {
    reference: string;
  };
  benchmark?: BenchmarkBlock;
  features: {
    morgan: { radius: number; n_bits: number };
    physicochemical: string[];
    maccs_bits: number;
  };
};
