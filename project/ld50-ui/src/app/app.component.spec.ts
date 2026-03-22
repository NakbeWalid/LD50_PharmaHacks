import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';
import { AppComponent } from './app.component';

const mockReport = {
  metrics: {
    description: 'test',
    split_rule: 'test',
    train: { n: 1, mae: 0, rmse: 0, r2: 0 },
    valid: { n: 1, mae: 0, rmse: 0, r2: 0 },
    test: { n: 1, mae: 0, rmse: 0, r2: 0 },
  },
  plots: {
    scatter_validation: { y_true: [1], y_pred: [1] },
    shap: null,
  },
  dataset: {
    split_sizes: { train: 1, valid: 1, test: 1 },
    sample_rows: [{ Drug_ID: 'x', Drug: 'C', Y: 1 }],
  },
  model: { name: 'XGB', params: { a: 1 } },
  benchmark: {
    description: 'test',
    r2_threshold: 0.6,
    svr_note: 'test',
    leaderboard: [
      {
        model: 'XGBoost',
        r2_validation: 0.5,
        mae_validation: 0.4,
        train_time_s: 1,
      },
    ],
    rank_by_model: { XGBoost: 1 },
  },
  features: {
    morgan: { radius: 2, n_bits: 1024 },
    physicochemical: ['MolWt'],
    maccs_bits: 167,
  },
};

describe('AppComponent', () => {
  let httpMock: HttpTestingController;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [AppComponent],
      providers: [provideHttpClient(), provideHttpClientTesting()],
    }).compileComponents();
    httpMock = TestBed.inject(HttpTestingController);
  });

  it('should create the app', () => {
    const fixture = TestBed.createComponent(AppComponent);
    fixture.detectChanges();
    httpMock.expectOne('/api/report').flush(mockReport);
    fixture.detectChanges();
    expect(fixture.componentInstance).toBeTruthy();
  });

  it('should render heading', () => {
    const fixture = TestBed.createComponent(AppComponent);
    fixture.detectChanges();
    httpMock.expectOne('/api/report').flush(mockReport);
    fixture.detectChanges();
    const compiled = fixture.nativeElement as HTMLElement;
    expect(compiled.querySelector('h1')?.textContent).toContain('LD50');
  });
});
