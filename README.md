# ESS 배터리 수명 예측 Mini Project

SKALA 3반 · U095 · 이연주

DAY 1 산출물: 제공된 3개 Batch의 EDA, 데이터 품질 점검, 초기 100사이클 기반 수명 회귀 설계이다. 모델 학습 및 성능 평가는 DAY 2 범위이다.

## 파일 안내

|파일|역할|
|---|---|
|`notebooks/30-ESSHealth-scratch.ipynb`|사용자가 실행한 교수님 기본 분석. 이번 제작에서 추가 수정하지 않음|
|`notebooks/01_data_check.ipynb`|구조·품질·정답·파일 지문 점검, 실행 결과 포함|
|`notebooks/02_day1_eda.ipynb`|다섯 질문 EDA와 쉬운 해설, 피처·모델·검증 전략|
|`src/day1_analysis.py`|선택적 HDF5 로더 및 분석·그림 함수|
|`src/day1_review.py`|배치별 분포·상대 ΔQ·전류 패턴·기울기 민감도·공선성 보완|
|`results/DS-MINI-Design-SKALA_3반-U095_이연주.pdf`|DAY 1 제출 보고서|
|`results/day1/figures/`|PNG 그래프 17개|
|`results/day1/tables/`|구조·셀 피처·제외 사유·상관·정책·검증 분리 표|
|`results/day1/*.html`|실행된 노트북의 읽기용 HTML|
|`results/day1/notebook_execution.json`, `verification.json`|실행 및 무결성 검증 기록|

## 실행

프로젝트 루트에서 `.venv`를 사용한다. 원본 `.mat` 3개는 기존 `data/`에 둔다.

```bash
source .venv/bin/activate
python -m pip install -r requirements-day1.txt
jupyter lab
```

`01_data_check.ipynb` → `02_day1_eda.ipynb` 순서로 커널을 재시작하고 모두 실행한다. 각각 독립적으로 데이터를 읽으며 전체 8GB를 메모리에 올리지 않는다. HDF5에서 summary와 필요한 초기 곡선만 읽는다. 첫 노트북의 SHA-256 계산은 원본 파일 전체를 읽는다.

자동 재생성(수정한 새 노트북은 덮어쓰므로 원본 생성 상태를 재현할 때 사용):

```bash
python scripts/execute_day1.py
python scripts/build_report.py
```

PDF 한글 폰트는 macOS AppleGothic을 사용한다. 다른 운영체제에서는 `ESS_KOREAN_FONT` 환경변수에 한글 TTF 경로를 지정한다. 분석 그래프의 영문 변수명은 코드와 직접 대응하도록 유지했다.

## 검증된 주요 결과

- 원본 139개 셀: Batch 1/2/3 = 46/47/46.
- 수명 유효 정답 129개. NaN 10개는 대치하지 않음.
- 종료 미확인·공개 코드 품질 경고 등을 표시한 주분석: 36/39/40 = 115개.
- Batch별 원본 유효 정답 중앙값: 858.5 / 472.0 / 1005.5 사이클.
- 주분석 `log10 Var(ΔQ)`와 수명의 Pearson r: -0.827 / -0.902 / -0.742.
- Batch 1에서 500 미만 셀은 0개: 분류보다 초기 100사이클 회귀 선택.
- Batch 1 정책 그룹 분리: train 29, validation 7. Train 안에서 4-fold GroupKFold 계획.

## 꼭 읽어야 하는 해석 범위

현재 Batch 2는 **2018-02-20**이다. 저자 공개 코드가 사용하는 2017-06-30과 달라 논문의 배치 병합을 적용하지 않았다. Batch 1 일부 기록은 후속 측정이 없으므로 전체 수명이 확인되지 않았다. 원본 분포와 품질 점검 후 분석을 함께 제공한다.

DAY 1 과제로 Batch 2·3의 정답 분포와 상관을 관찰했다. DAY 2 평가는 EDA에서 관찰한 외부 Batch 평가이지 완전한 블라인드 테스트가 아니다. 새로운 미관찰 배치로 추가 검증해야 한다. 논문 9.1%를 이번 모델 성능으로 주장하지 않는다.

코드의 `analysis_eligible`은 이번 분석의 명시적 포함 기준이다. 실제 배포에서는 결측·미완료 셀의 선택 편향과 제외 규칙의 민감도를 추가 검토해야 한다.

## 최종 검토에서 보완한 내용

- 핵심 피처 6개의 배치별 분포, 세 Batch의 공선성, 상관 상위 피처를 추가했다.
- <500 셀이 없는 Batch는 수명 하위·상위 25%의 상대 ΔQ 비교로 보완했다. 절대 단수명 정의를 변경한 것은 아니다.
- 최단 셀의 Batch 내 백분위와 동일 정책의 다른 셀을 비교했다. 인과 원인은 확정하지 않는다.
- cycle 10의 실제 충전 전류 패턴과 초기 용량 기울기의 평활화 민감도를 점검했다.
- 동일 수명 셀의 대표 선택을 셀 ID로 고정해 보고서 표와 그림을 일치시켰다.
- 새 노트북·보고서 문체를 '~다'로 통일했다. Scratch는 보존했다.

### Scratch를 그대로 두는 이유

새 노트북은 Scratch의 코드·변수·출력 CSV에 의존하지 않고 MAT 원본에서 직접 계산한다. Scratch의 셀 순서 기반 색상을 수명 색상으로 읽거나, EOL 아래 용량을 이상치로 제거하면 해석이 달라질 수 있다. 기존 Scratch는 수업 참고용으로 유지하고 제출 분석에서는 실제 수명 색상, 넓은 물리적 품질 범위, 원래 사이클 번호, 셀당 1행을 사용한다. 교수님 기본 로더의 `int(cycle_life)`를 결측 수명이 있는 다른 Batch에 그대로 적용하지 않는다.

## 제출 및 학습

PDF와 노트북 결과를 먼저 확인하고, Q1 → Q3 → Q5 → 모델 전략 순서로 읽으면 흐름을 파악하기 쉽다. 보고서 표지·바닥글에 SKALA 3반 U095 이연주를 반영했다. 제출처 업로드는 별도로 진행한다.

AI 도구를 사용해 분석 코드·문서 초안을 작성하고 실행 결과로 검증했다. 최종 제출 전 수치와 해석을 직접 이해하고 과제의 AI 사용 규칙을 확인한다.

## 참고문헌

- Severson et al. (2019), *Data-driven prediction of battery cycle life before capacity degradation*, Nature Energy 4, 383–391. https://doi.org/10.1038/s41560-019-0356-8
- 저자 공개 처리 코드: https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation
- 교수님 제공 Scratch·강의 자료·DAY 1 과제 화면.
