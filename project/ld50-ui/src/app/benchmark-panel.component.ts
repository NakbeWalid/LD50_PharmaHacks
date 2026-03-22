import { DecimalPipe } from '@angular/common';
import {
  AfterViewInit,
  Component,
  ElementRef,
  Input,
  OnDestroy,
  ViewChild,
} from '@angular/core';
import { Chart, registerables } from 'chart.js';
import annotationPlugin from 'chartjs-plugin-annotation';
import type { BenchmarkBlock } from './report.types';

Chart.register(...registerables, annotationPlugin);

const AXIS = {
  ticks: { color: '#ffffff' },
  grid: { color: 'rgba(255, 255, 255, 0.14)' },
  border: { color: 'rgba(255, 255, 255, 0.22)' },
} as const;

@Component({
  selector: 'app-benchmark-panel',
  standalone: true,
  imports: [DecimalPipe],
  template: `
    <div class="bench-wrap">
      <h3>Model benchmark (CELL 7.5 — validation set)</h3>
      <p class="hint">{{ benchmark.description }}</p>
      <p class="hint svr">{{ benchmark.svr_note }}</p>
      <p class="hint">
        Presentation target from the notebook: R² ≥ {{ benchmark.r2_threshold }} (dashed line in the original seaborn plot).
      </p>

      <div class="table-wrap">
        <table class="tbl">
          <thead>
            <tr>
              <th>Rank</th>
              <th>Model</th>
              <th class="num">R² (valid)</th>
              <th class="num">MAE (valid)</th>
              <th class="num">Time (s)</th>
            </tr>
          </thead>
          <tbody>
            @for (row of benchmark.leaderboard; track row.model) {
              <tr>
                <td>{{ benchmark.rank_by_model[row.model] }}</td>
                <td>
                  {{ row.model }}
                  @if (row.train_rows_used) {
                    <span class="sub"> (train n={{ row.train_rows_used }})</span>
                  }
                </td>
                <td class="num">{{ row.r2_validation | number : '1.4-4' }}</td>
                <td class="num">{{ row.mae_validation | number : '1.4-4' }}</td>
                <td class="num">{{ row.train_time_s | number : '1.2-2' }}</td>
              </tr>
            }
          </tbody>
        </table>
      </div>

      <p class="chart-title">R² on validation (higher is better)</p>
      <div class="canvas-wrap">
        <canvas #bar></canvas>
      </div>
    </div>
  `,
  styles: [
    `
      .bench-wrap {
        margin-top: 0.5rem;
      }
      .bench-wrap h3 {
        margin: 0 0 0.5rem;
        font-size: 1rem;
        font-weight: 650;
      }
      .hint {
        margin: 0 0 0.5rem;
        font-size: 0.82rem;
        color: var(--muted);
        line-height: 1.4;
      }
      .hint.svr {
        font-size: 0.78rem;
        opacity: 0.95;
      }
      .chart-title {
        margin: 1rem 0 0.35rem;
        font-size: 0.85rem;
        color: #cbd5e1;
      }
      .canvas-wrap {
        position: relative;
        height: 220px;
        max-width: 40rem;
      }
      .tbl .sub {
        font-size: 0.75rem;
        color: var(--muted);
      }
    `,
  ],
})
export class BenchmarkPanelComponent implements AfterViewInit, OnDestroy {
  @Input({ required: true }) benchmark!: BenchmarkBlock;

  @ViewChild('bar') private barRef?: ElementRef<HTMLCanvasElement>;

  private chart?: Chart;

  ngAfterViewInit(): void {
    setTimeout(() => this.draw(), 0);
  }

  ngOnDestroy(): void {
    this.chart?.destroy();
  }

  private draw(): void {
    const canvas = this.barRef?.nativeElement;
    if (!canvas) return;

    const lb = [...this.benchmark.leaderboard].sort(
      (a, b) => b.r2_validation - a.r2_validation,
    );
    const labels = lb.map((r) => r.model);
    const vals = lb.map((r) => r.r2_validation);
    const th = this.benchmark.r2_threshold;
    const xmax = Math.min(0.9, Math.max(0.75, ...vals, th) + 0.05);

    this.chart?.destroy();
    this.chart = new Chart(canvas, {
      type: 'bar',
      data: {
        labels,
        datasets: [
          {
            label: 'R² (validation)',
            data: vals,
            backgroundColor: vals.map((v) =>
              v >= th ? 'rgba(52, 211, 153, 0.8)' : 'rgba(251, 191, 36, 0.85)',
            ),
          },
        ],
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            labels: { color: '#ffffff' },
          },
          annotation: {
            annotations: {
              r2Threshold: {
                type: 'line',
                scaleID: 'x',
                value: th,
                borderColor: 'rgba(248, 113, 113, 0.95)',
                borderWidth: 2,
                borderDash: [6, 4],
                label: {
                  display: true,
                  content: `Target R² = ${th.toFixed(2)}`,
                  color: '#fef2f2',
                  backgroundColor: 'rgba(127, 29, 29, 0.85)',
                  padding: 4,
                  borderRadius: 4,
                  position: 'start',
                  rotation: 90,
                  font: { size: 10, weight: '600' },
                },
              },
            },
          },
        },
        scales: {
          x: {
            ...AXIS,
            min: 0,
            max: xmax,
            title: { display: true, text: 'R² (validation)', color: '#ffffff' },
          },
          y: { ...AXIS },
        },
      },
    } as any);
  }
}
