import { DecimalPipe } from '@angular/common';
import {
  AfterViewInit,
  Component,
  effect,
  ElementRef,
  HostListener,
  Input,
  OnDestroy,
  signal,
  ViewChild,
} from '@angular/core';
import { Chart, registerables } from 'chart.js';
import type { Report } from './report.types';

Chart.register(...registerables);

const AXIS = {
  ticks: { color: '#ffffff' },
  grid: { color: 'rgba(255, 255, 255, 0.14)' },
  border: { color: 'rgba(255, 255, 255, 0.22)' },
} as const;

function scaleTitle(text: string) {
  return {
    display: true,
    text,
    color: '#ffffff',
    font: { size: 12, weight: 500 as const },
  };
}

const LEGEND_WHITE = {
  labels: { color: '#ffffff', font: { size: 11 } },
};

@Component({
  selector: 'app-chart-panel',
  standalone: true,
  template: `
    @if (focusId()) {
      <div
        class="chart-backdrop"
        role="presentation"
        (click)="clearFocus()"
        aria-hidden="true"
      ></div>
    }
    <div class="chart-grid">
      <div class="chart-slot">
        <div
          class="chart-card"
          [class.expanded]="focusId() === 'scatterValid'"
          (click)="toggleFocus('scatterValid', $event)"
        >
          <h3>XGBoost Model Performance: Predictions vs Ground Truth (Validation)</h3>
          <p class="chart-hint">
            Points near the gold dashed line (ideal y = x) are better.
          </p>
          <div class="canvas-wrap">
            <canvas #scatterValid></canvas>
          </div>
        </div>
      </div>
      @if (report.plots.shap; as sh) {
        <div class="chart-slot">
          <div
            class="chart-card"
            [class.expanded]="focusId() === 'shapBar'"
            (click)="toggleFocus('shapBar', $event)"
          >
          <h3>What makes a molecule toxic? (Top 10 Features)</h3>
            <div class="canvas-wrap tall">
              <canvas #shapBar></canvas>
            </div>
          </div>
        </div>
        <div class="chart-slot wide-slot">
          <div
            class="chart-card wide"
            [class.expanded]="focusId() === 'shapWf'"
            (click)="toggleFocus('shapWf', $event)"
          >
          <h3>Prediction Explanation for: {{ sh.waterfall.molecule_id }}</h3>
          <p class="chart-hint">
            Validation index {{ sh.waterfall.molecule_index }} · predicted
            {{ sh.waterfall.predicted_log_ld50 | number : '1.3-3' }} · true
            {{ sh.waterfall.actual_log_ld50 | number : '1.3-3' }}
          </p>
            <div class="canvas-wrap">
              <canvas #shapWaterfall></canvas>
            </div>
          </div>
        </div>
      }
    </div>
  `,
  styles: [
    `
      .chart-backdrop {
        position: fixed;
        inset: 0;
        z-index: 900;
        background: transparent;
      }
      .chart-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(min(100%, 22rem), 1fr));
        gap: 1.25rem;
        margin-top: 1rem;
        position: relative;
        z-index: 0;
      }
      .chart-slot {
        min-height: 340px;
      }
      .chart-slot.wide-slot {
        grid-column: 1 / -1;
        min-height: 400px;
      }
      .chart-card {
        position: relative;
        z-index: 1;
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem 1.1rem 1.25rem;
        box-shadow: var(--shadow);
        cursor: zoom-in;
        transition:
          box-shadow 0.2s ease,
          transform 0.2s ease;
      }
      .chart-card:hover:not(.expanded) {
        border-color: rgba(56, 189, 248, 0.35);
        box-shadow: 0 12px 36px rgba(0, 0, 0, 0.35);
      }
      .chart-card.expanded {
        position: fixed;
        z-index: 950;
        left: 50%;
        top: 50%;
        transform: translate(-50%, -50%);
        width: min(92vw, 900px);
        max-height: 90vh;
        overflow: auto;
        cursor: zoom-out;
        box-shadow: 0 28px 90px rgba(0, 0, 0, 0.55);
        border-color: rgba(56, 189, 248, 0.45);
        outline: 1px solid rgba(56, 189, 248, 0.25);
      }
      .chart-card h3 {
        margin: 0 0 0.35rem;
        font-size: 1rem;
        font-weight: 650;
        color: var(--text);
      }
      .chart-hint {
        margin: 0 0 0.75rem;
        font-size: 0.8rem;
        color: var(--muted);
      }
      .canvas-wrap {
        position: relative;
        height: 260px;
      }
      .canvas-wrap.tall {
        height: 320px;
      }
      .chart-card.expanded .canvas-wrap {
        height: min(52vh, 460px);
      }
      .chart-card.expanded .canvas-wrap.tall {
        height: min(58vh, 520px);
      }
    `,
  ],
  imports: [DecimalPipe],
})
export class ChartPanelComponent implements AfterViewInit, OnDestroy {
  @Input({ required: true }) report!: Report;

  readonly focusId = signal<string | null>(null);

  @ViewChild('scatterValid') private svRef?: ElementRef<HTMLCanvasElement>;
  @ViewChild('shapBar') private shapBarRef?: ElementRef<HTMLCanvasElement>;
  @ViewChild('shapWaterfall') private shapWfRef?: ElementRef<HTMLCanvasElement>;

  private charts: Chart[] = [];

  constructor() {
    effect(() => {
      this.focusId();
      setTimeout(() => this.charts.forEach((c) => c.resize()), 120);
    });
  }

  @HostListener('document:keydown.escape')
  onEscape(): void {
    if (this.focusId()) {
      this.focusId.set(null);
    }
  }

  ngAfterViewInit(): void {
    setTimeout(() => this.buildAll(), 0);
  }

  ngOnDestroy(): void {
    this.destroyCharts();
  }

  toggleFocus(id: string, ev: Event): void {
    ev.stopPropagation();
    this.focusId.update((f) => (f === id ? null : id));
  }

  clearFocus(): void {
    this.focusId.set(null);
  }

  private destroyCharts(): void {
    this.charts.forEach((c) => c.destroy());
    this.charts = [];
  }

  private buildAll(): void {
    this.destroyCharts();
    const r = this.report;

    this.scatterPlot(this.svRef?.nativeElement, r.plots.scatter_validation);

    const shap = r.plots.shap;
    if (shap && this.shapBarRef?.nativeElement) {
      const g = shap.global_top10.slice().reverse();
      this.charts.push(
        new Chart(this.shapBarRef.nativeElement, {
          type: 'bar',
          data: {
            labels: g.map((x) => x.feature),
            datasets: [
              {
                label: 'Mean |SHAP|',
                data: g.map((x) => x.mean_abs_shap),
                backgroundColor: 'rgba(167, 139, 250, 0.92)',
                borderColor: 'rgba(255, 255, 255, 0.35)',
                borderWidth: 1,
              },
            ],
          },
          options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
              x: {
                ...AXIS,
                beginAtZero: true,
                title: scaleTitle('Mean |SHAP|'),
              },
              y: { ...AXIS },
            },
          },
        }),
      );
    }

    if (shap && this.shapWfRef?.nativeElement) {
      const wf = shap.waterfall.features;
      this.charts.push(
        new Chart(this.shapWfRef.nativeElement, {
          type: 'bar',
          data: {
            labels: wf.map((f) => f.feature),
            datasets: [
              {
                label: 'SHAP value',
                data: wf.map((f) => f.shap),
                backgroundColor: wf.map((f) =>
                  f.shap >= 0
                    ? 'rgba(252, 165, 165, 0.95)'
                    : 'rgba(147, 197, 253, 0.95)',
                ),
                borderColor: wf.map((f) =>
                  f.shap >= 0 ? 'rgba(254, 202, 202, 0.9)' : 'rgba(191, 219, 254, 0.9)',
                ),
                borderWidth: 1,
              },
            ],
          },
          options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
              x: {
                ...AXIS,
                title: scaleTitle('SHAP (impact on prediction)'),
              },
              y: { ...AXIS },
            },
          },
        }),
      );
    }
  }

  private scatterPlot(
    canvas: HTMLCanvasElement | undefined,
    data: { y_true: number[]; y_pred: number[] },
  ): void {
    if (!canvas) return;
    const yt = data.y_true;
    const yp = data.y_pred;
    const pts = yt.map((x, i) => ({ x, y: yp[i]! }));
    const lo = Math.min(...yt, ...yp);
    const hi = Math.max(...yt, ...yp);

    const scatterOpts = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'bottom' as const, ...LEGEND_WHITE },
      },
      scales: {
        x: {
          ...AXIS,
          title: scaleTitle('True log(LD50)'),
        },
        y: {
          ...AXIS,
          title: scaleTitle('Predicted log(LD50)'),
        },
      },
    };

    this.charts.push(
      new Chart(canvas, {
        type: 'scatter',
        data: {
          datasets: [
            {
              type: 'scatter',
              label: 'Molecules',
              data: pts,
              backgroundColor: 'rgba(56, 189, 248, 0.92)',
              borderColor: 'rgba(255, 255, 255, 0.95)',
              borderWidth: 1.25,
              pointRadius: 4.5,
              pointHoverRadius: 6,
            },
            {
              type: 'line',
              label: 'Ideal (y = x)',
              data: [
                { x: lo, y: lo },
                { x: hi, y: hi },
              ],
              borderColor: 'rgb(250, 204, 21)',
              borderDash: [7, 5],
              borderWidth: 2.5,
              pointRadius: 0,
              fill: false,
            },
          ],
        },
        options: scatterOpts,
      } as any),
    );
  }
}
