import { DecimalPipe } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { Component, inject, OnInit, signal } from '@angular/core';

type SplitMetrics = {
  n: number;
  mae: number;
  rmse: number;
  r2: number;
};

type MetricsResponse = {
  description?: string;
  split_rule?: string;
  train: SplitMetrics;
  valid: SplitMetrics;
  test: SplitMetrics;
};

@Component({
  selector: 'app-root',
  imports: [DecimalPipe],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css',
})
export class AppComponent implements OnInit {
  private readonly http = inject(HttpClient);

  /**
   * En dev (`ng serve`), les appels passent par le proxy Angular : `/api` → backend
   * (voir `proxy.conf.json`, champ `target` = URL uvicorn).
   */
  readonly apiBase = '/api';

  readonly rows = signal<
    { label: string; key: 'train' | 'valid' | 'test' }[]
  >([
    { label: 'Train', key: 'train' },
    { label: 'Validation', key: 'valid' },
    { label: 'Test', key: 'test' },
  ]);

  metrics = signal<MetricsResponse | null>(null);
  error = signal<string | null>(null);
  loading = signal(true);

  ngOnInit(): void {
    this.loadMetrics();
  }

  loadMetrics(): void {
    this.error.set(null);
    this.loading.set(true);
    this.http.get<MetricsResponse>(`${this.apiBase}/metrics`).subscribe({
      next: (m) => {
        this.metrics.set(m);
        this.loading.set(false);
      },
      error: (e) => {
        this.loading.set(false);
        const msg =
          e?.error?.detail ??
          e?.message ??
          'Erreur réseau — lance uvicorn, vérifie proxy.conf.json (target = même port), exécute train_and_save.py (voir LOCAL_WEB_LD50.md).';
        this.error.set(typeof msg === 'string' ? msg : JSON.stringify(msg));
      },
    });
  }
}
