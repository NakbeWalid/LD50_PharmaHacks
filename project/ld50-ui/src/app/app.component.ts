import { DecimalPipe } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { Component, inject, OnInit, signal } from '@angular/core';
import { BenchmarkPanelComponent } from './benchmark-panel.component';
import { ChartPanelComponent } from './chart-panel.component';
import { NB } from './notebook-copy';
import { PARAM_HINTS, PARAM_ORDER } from './model-param-hints';
import type { Report } from './report.types';

@Component({
  selector: 'app-root',
  imports: [DecimalPipe, BenchmarkPanelComponent, ChartPanelComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css',
})
export class AppComponent implements OnInit {
  private readonly http = inject(HttpClient);

  readonly nb = NB;

  readonly apiBase = '/api';

  readonly rows = signal<
    { label: string; key: 'train' | 'valid' | 'test' }[]
  >([
    { label: 'Train', key: 'train' },
    { label: 'Validation', key: 'valid' },
    { label: 'Test', key: 'test' },
  ]);

  report = signal<Report | null>(null);
  error = signal<string | null>(null);
  loading = signal(true);

  ngOnInit(): void {
    this.loadReport();
  }

  paramsWithHints(params: Record<string, string | number>): {
    key: string;
    value: string | number;
    hint: string;
  }[] {
    const seen = new Set<string>();
    const rows: { key: string; value: string | number; hint: string }[] = [];
    for (const k of PARAM_ORDER) {
      if (k in params) {
        rows.push({
          key: k,
          value: params[k]!,
          hint: PARAM_HINTS[k] ?? '',
        });
        seen.add(k);
      }
    }
    for (const k of Object.keys(params).sort()) {
      if (!seen.has(k)) {
        rows.push({
          key: k,
          value: params[k]!,
          hint: PARAM_HINTS[k] ?? '',
        });
      }
    }
    return rows;
  }

  loadReport(): void {
    this.error.set(null);
    this.loading.set(true);
    this.http.get<Report>(`${this.apiBase}/report`).subscribe({
      next: (r) => {
        this.report.set(r);
        this.loading.set(false);
      },
      error: (e) => {
        this.loading.set(false);
        const msg =
          e?.error?.detail ??
          e?.message ??
          'Network error — start uvicorn, check proxy target, run train_and_save.py (see LOCAL_WEB_LD50.md).';
        this.error.set(typeof msg === 'string' ? msg : JSON.stringify(msg));
      },
    });
  }
}
