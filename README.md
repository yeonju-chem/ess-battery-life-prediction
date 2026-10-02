# ESS 배터리 수명 예측

초기 100사이클의 방전 데이터로 배터리 셀의 전체 Cycle Life를 예측하고, 다른 실험 배치에서도 성능이 유지되는지 평가한다.

## 프로젝트 개요

- 데이터셋: MIT-Stanford Battery Dataset 계열 제공 파일 (Severson et al., 2019)
- 학습 데이터: Batch 1 (2017-05-12), 품질 점검 후 36셀
- 평가 데이터: Batch 2 (2018-02-20), 39셀; 추가 평가 Batch 3 (2018-04-12), 40셀
- 태스크: Regression (Cycle Life 예측), 주 지표 MAPE
- 원본 139셀 중 정답·품질 기준을 충족한 115셀을 분석했다. 제공된 Batch 2 파일은 원논문 공개 코드의 배치 날짜와 달라 논문 실험의 동일 재현은 아니다.

## 파일 구조

```text
├── data/                         # 원본 MAT 파일 3개와 안내
├── notebooks/
│   ├── 30-ESSHealth-scratch.ipynb # 교수 제공 탐색 노트북
│   ├── 01_data_check.ipynb        # 구조·품질 확인
│   ├── 02_day1_eda.ipynb          # EDA·모델 전략
│   └── 03_day2_modeling.ipynb     # 모델 학습·평가, 실행 결과 포함
├── src/                          # 분석·피처·모델 구현
├── results/
│   ├── DS-MINI-Design-SKALA_3반-U095_이연주.pdf
│   ├── DS-MINI-Model-SKALA_3반-U095_이연주.pdf
│   ├── day1/                      # EDA 표·그림
│   └── day2/                      # 성능표·예측값·그림·저장 모델
├── requirements.txt
└── README.md
```

## 환경 설정

```bash
git clone https://github.com/yeonju-chem/ess-battery-life-prediction.git
cd ess-battery-life-prediction
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

`data/README.md`에 따라 MAT 파일 3개를 `data/`에 둔다. 노트북은 `01_data_check.ipynb` → `02_day1_eda.ipynb` → `03_day2_modeling.ipynb` 순서로 실행한다.

## EDA

- **Cycle Life 분포:** Batch 2는 500사이클 미만이 28/39셀(71.8%)이고, Batch 3는 1,000사이클 초과가 23/44셀(52.3%)이다. 배치별 수명 분포가 달라 성능을 나누어 평가했다.
- **열화 곡선:** 여러 셀에서 후기 방전 용량 감소가 빨라지는 knee 후보가 보인다. 전체 수명을 본 뒤 찾은 값이므로 초기 예측 입력에서 제외했다.
- **ΔQ(V):** 사이클 100과 10의 방전 곡선 차이 `Q100(V)-Q10(V)`를 계산했다. 그 로그분산은 세 Batch 모두 수명과 음의 상관을 보여 핵심 피처로 선정했다.
- **충전 조건:** 정책별 평균 수명은 다르지만 셀 수가 적고 Batch·구조 차이가 섞여 있다. 빠른 충전이 수명을 단축한다는 단일 인과 결론은 내리지 않았다.
- **피처 중복:** ΔQ 분산과 표준편차, ΔQ 평균과 면적은 강하게 중복되어 대표값만 우선 사용했다.

## Modeling

### 피처 엔지니어링 전략

F1은 `log_dq_var` 단독이다. F2는 초기 Qd 기울기와 등가 C-rate, F3는 온도와 내부저항을 추가한다. 세 피처 세트의 추가 가치는 Batch 1 내부 정책 그룹 CV에서 비교했다. 후기 knee·마지막 용량·전체 기록 길이처럼 예측 시점에 알 수 없는 정보는 제외했다.

### 모델 선택 및 근거

- 후보 모델: 중앙값 기준모델, Ridge, ElasticNet, 얕은 Random Forest
- 최종 모델: ElasticNet + F1, 원값 타깃, `alpha=0.1`, `l1_ratio=0.8`
- 선택 이유: 29셀의 4-fold 정책 그룹 CV에서 MAPE가 가장 낮았다. Ridge와 차이는 매우 작아 ElasticNet의 일반적 우월성을 뜻하지 않는다. 결측 대치·표준화는 각 학습 fold 안에서 적합했다.

## 성능 결과

| 구분                     | MAPE / Gap |
| ------------------------ | ---------: |
| Train (Batch 1 CV)       |      9.21% |
| Valid (Batch 1 Hold-out) |      9.20% |
| Test (Batch 2)           |     26.21% |
| Gap (Train-Valid)        |    -0.01%p |
| Gap (Valid-Test)         |   +17.01%p |
| Gap (Target-Test)        |   +17.11%p |
| Test (Batch 3, 추가)     |     12.16% |

Gap은 각각 Valid−CV, Batch 2−Valid, Batch 2−원논문 기준 9.1%다. Batch 2 MAPE는 목표보다 17.11%p 높다. [필수 성능표](results/day2/tables/model_performance.csv)와 [Batch 3 추가 성능표](results/day2/tables/model_performance_extended.csv)에 세부 결과가 있다.

## 오류 분석

Batch 2의 500사이클 미만 셀 28개는 평균 약 130사이클 과대예측했다. 오차가 가장 큰 `b2c6`은 실제 393사이클, 예측 653.50사이클(APE 66.28%)이다. Batch 1 학습 범위가 534사이클부터 시작해 짧은 수명 영역을 충분히 배우지 못한 것이 한 가지 원인 가설이다. Batch 3의 장수명 셀은 반대로 과소예측하는 경향이 남았다. 단수명·새 정책의 학습 데이터 확보와 독립 배치 재검증이 필요하다.

## ESS 도메인 해석

초기 수명 예측은 셀 선별과 교체 우선순위를 검토하는 보조 정보로 활용할 수 있다. 현재 모델은 Batch 2 단수명을 과대예측하므로 교체 시점을 단독으로 결정하는 데 쓰기 어렵다. 실험실 셀과 실제 BESS의 온도·부하·달력 열화 조건도 다르다. 실제 적용 전에는 현장 데이터 검증, 예측 오차·입력 분포 변화 모니터링, 안전 기준이 필요하다.

## 참고문헌

- Severson, K. A., et al. (2019). Data-driven prediction of battery cycle life before capacity degradation. _Nature Energy_, 4, 383–391. https://doi.org/10.1038/s41560-019-0356-8

## 팀 구성

- SKALA 울산 3반 U095 이연주: EDA, 피처 엔지니어링, 모델 개발, Batch 2·3 성능 평가
