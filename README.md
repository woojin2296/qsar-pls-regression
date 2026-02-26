# qsar-pls-regression

`qsar-pls-regression`은 PLS(Partial Least Squares) 회귀를 사용해 Mpro / RdRp 분자 데이터셋을 비교하는 Jupyter 실험 모음입니다.

## 이 프로젝트가 할 수 있는 것 (What this project can do)

- **PLS 회귀 모델 학습**: 분자 특성(descriptor) 데이터를 입력으로 받아 결합 에너지(binding energy)를 예측하는 PLS 회귀 모델을 학습합니다.
- **다양한 데이터셋 지원**: Mpro(665개), RdRp(29, 76, 240개) 등 여러 크기의 분자 데이터셋에 대해 실험을 수행합니다.
- **Train/Test 분할 평가**: 데이터를 학습/테스트 세트로 분할하고 RMSE, R², MSE, RMRPE 등의 지표로 모델 성능을 평가합니다.
- **교차 검증(Cross-Validation)**: k-fold 교차 검증(기본 cv=5)으로 모델의 일반화 성능을 평가합니다.
- **진단 플롯 생성**: Actual vs. Predicted, Parity Plot, Residuals, Bland-Altman, Hexbin Density, PLS Score Plot, Feature Importance 등 다양한 시각화 결과물을 PNG 파일로 저장합니다.

## 폴더 구성

- `Mpro-pls-665-traintest.ipynb`
- `RdRp-pls-240-traintest.ipynb`
- `RdRp-pls-240-cv10.ipynb`
- `RdRp-pls-29-traintest.ipynb`
- `RdRp-pls-29-cv10.ipynb`
- `RdRp-pls-76-traintest.ipynb`
- `RdRp-pls-76-cv10.ipynb`
- `data/`: 실험 입력 CSV
- `plots_240_traintest/`: 일부 노트북이 저장하는 그림 폴더

## 데이터셋 요약 (`data/`)

| 파일 | 샘플 수 | 컬럼 수 | 타깃(y) | 비고 |
|---|---:|---:|---|---|
| `665-molecular-Mpro-data.csv` | 665 | 16 | `binding_energy` | 식별자: `DrugName`, 목표 제외 후 특성 사용 |
| `29-molecular-RdRp-data.csv` | 29 | 16 | `Binding energy(kcal/mol)` | 식별자: `Drug`, 목표 제외 후 특성 사용 |
| `76-molecular-RdRp-data.csv` | 76 | 16 | `RdRp Binding energy(kcal/mol)` | 식별자: `Drug`, Mpro/ RdRp 타깃 공존 |
| `240-molecular-RdRp-data.csv` | 239 | 16 | `best_affinity_kcal/mol` (cv10) / `binding_energy` (traintest) | 240-cv10은 `Drug`, `SMILES (pubchem)` 제거 |
| `76-molecular-Mpro-data.csv` | 76 | 16 | 이 폴더에 존재하지만 현재 노트북에서 직접 사용되지 않음 | `Mpro` 전용 특성명 규칙 사용 |

## 노트북별 실험 설정

- 공통 전처리
  - `pandas`로 CSV 로드
  - 타깃(`y`)과 특징(`X`) 분리
  - `X = data.drop(columns=[...])`, `y = data[target]`
  - 모델: `PLSRegression(n_components=10)`
  - 지표: `RMSE`, `R^2`(일부), 필요 시 `mse`, `RMRPE`

- `Mpro-pls-665-traintest.ipynb`
  - 데이터: `data/665-molecular-Mpro-data.csv`
  - 분할: `train_test_split(test_size=0.2, random_state=42)`
  - 출력: RMSE (`0.8027795779418136`) 및 여러 진단 플롯 저장

- `RdRp-pls-240-traintest.ipynb`
  - 데이터: `data/240-molecular-RdRp-data.csv`
  - 분할: `train_test_split(test_size=0.2, random_state=42)`
  - 출력: MSE/RMSE/R²/RMRPE 계산 및 진단 플롯 저장

- `RdRp-pls-240-cv10.ipynb`
  - 데이터: `data/240-molecular-RdRp-data.csv`
  - 타깃: `best_affinity_kcal/mol`
  - CV: `cross_val_score(..., cv=5, scoring='neg_root_mean_squared_error')`
  - 출력: `Cross-validated RMSE scores: 0.5508963593504496`

- `RdRp-pls-29-traintest.ipynb`
  - 데이터: `data/29-molecular-RdRp-data.csv`
  - 분할: `train_test_split(test_size=10, random_state=42)` (정수 개수 기준 split)
  - 출력: RMSE (`0.9298119345900049`) 및 scatter만 출력 (파일 저장 없음)

- `RdRp-pls-29-cv10.ipynb`
  - 데이터: `data/29-molecular-RdRp-data.csv`
  - CV: `cv=5`, scoring=`neg_root_mean_squared_error`
  - 출력: `Cross-validated RMSE scores: 0.9607344145005262`

- `RdRp-pls-76-traintest.ipynb`
  - 데이터: `data/76-molecular-RdRp-data.csv`
  - 타깃: `RdRp Binding energy(kcal/mol)` (또한 `Mpro Binding energy(kcal/mol)` 동시 존재)
  - 분할: `train_test_split(test_size=0.2, random_state=42)`
  - 출력: RMSE (`0.7915680419677422`) 및 scatter만 출력

- `RdRp-pls-76-cv10.ipynb`
  - 데이터: `data/76-molecular-RdRp-data.csv`
  - 분할: `cross_val_score(..., cv=5)`
  - 출력: `Cross-validated RMSE scores: 1.1686712743092313`

## 생성되는 주요 플롯 (`plots_240_traintest/`)

`Mpro-pls-665-traintest.ipynb`, `RdRp-pls-240-traintest.ipynb`에서 아래 파일명으로 저장:

- `actual_vs_pred_test.png`
- `parity_train_test.png`
- `residuals_vs_pred_test.png`
- `residual_distribution_test.png`
- `absolute_error_cdf_test.png`
- `hexbin_density_test.png`
- `bland_altman_test.png`
- `pls_score_pc1_pc2_train.png`
- `top_features_pls.png`

## 설치/실행 가이드

- Python 패키지
  - `pandas`, `numpy`, `scikit-learn`, `matplotlib`
  - `pip install pandas numpy scikit-learn matplotlib`

- 실행
  - `jupyter notebook`로 열고 각 노트북을 위에서 아래로 실행
  - 상대 경로 기준으로 `data/` 읽도록 구성되어 있음 (`input_file = 'data/....csv'`)

## 참고 및 주의점

- `240` 계열 데이터는 노트북별로 타깃 컬럼명이 다릅니다(`best_affinity_kcal/mol` vs `binding_energy`).
- 일부 노트북은 `test_size=10`(정수)로 분할하고, 일부는 `0.2` 비율 사용합니다.
- `plots_240_traintest`는 여러 노트북에서 공유하므로 재실행 시 기존 결과가 덮어써질 수 있습니다.
