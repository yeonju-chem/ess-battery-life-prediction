"""Generate readable DAY 1 notebooks. Run from the project root."""
from pathlib import Path
import json
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
def md(s):return nbf.v4.new_markdown_cell(s.strip())
def code(s):return nbf.v4.new_code_cell(s.strip())
setup='''from pathlib import Path
import sys, os
ROOT = Path(os.environ.get("ESS_PROJECT_ROOT", Path.cwd())).resolve()
if ROOT.name == "notebooks": ROOT = ROOT.parent
assert (ROOT / "data").exists(), "프로젝트 루트에서 실행한다."
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
import day1_analysis as da
import day1_review as review
da.configure()
OUT = ROOT / "results" / "day1"
(OUT / "tables").mkdir(parents=True, exist_ok=True)
(OUT / "figures").mkdir(parents=True, exist_ok=True)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 90)
print("프로젝트:", ROOT)
print("random_state =", da.RANDOM_STATE)'''
common='''cells, manifest = da.load_batches(ROOT / "data")
features, quality = da.feature_table(cells)
summary, corr = da.make_tables(cells, features, quality, manifest, OUT)
display(summary.round(3))'''
nb1=[
md('''# 01. 데이터 구조와 품질 확인
SKALA 3반 · U095 · 이연주 | DAY 1 | 2026-10-01

세 개 `.mat` 파일을 직접 읽어 **배터리 셀 단위**로 정리한다. 원본 데이터는 수정하지 않는다.
Scratch의 `batch → cell → summary / cycles` 구조를 그대로 사용하되, h5py로 필요한 배열만 읽는다.
이는 mat73의 미지원 문자열 경고와 전체 시계열 로딩 비용을 피하기 위한 구현 선택이다.

**실행:** 프로젝트의 `.venv` 커널 선택 → Restart Kernel → Run All.
이 노트북은 모델을 학습하지 않는다. CSV는 UTF-8 BOM 형식이다.
셀 ID는 파일 내 0부터 시작하는 인덱스이며, 예를 들어 `b1c0`은 Batch 1의 첫 셀이다.'''),
code(setup),md('''## 1. 데이터 출처와 구조
Batch 1 = 2017-05-12, Batch 2 = **2018-02-20**, Batch 3 = 2018-04-12.
논문 공개 코드의 Batch 2는 2017-06-30이다. 따라서 그 코드의 Batch 1/2 병합 규칙을 이 파일들에 적용하지 않는다.

`summary`: 사이클별 QD·QC·IR·온도·충전시간·실제 cycle 번호.
`cycles`: 사이클 내부의 전압·전류·Qdlin 등. `Vdlin`: Qdlin의 공통 전압축.
`cycle_life`: 제공 파일의 정답 필드. 누락되거나 종료가 불명확하면 임의로 보충하지 않는다.'''),
code(common),code('''display(manifest)
print("첫 셀 summary 자료형")
display(cells[0]["summary"].dtypes.to_frame("dtype"))
display(cells[0]["summary"].head())
for b in da.FILES:
    c = next(c for c in cells if c["batch"] == b)
    print(b, "Vdlin:", len(c["voltage"]), "전압 범위:", c["voltage"].min(), c["voltage"].max())
    print("Qdlin cycle 10/100:", c["qcurves"][10].shape, c["qcurves"][100].shape)'''),
md('''## 2. 결측과 0은 다르다
summary의 NaN 수가 0이어도 첫 사이클의 0, 과도한 충전시간, 비정상 용량은 남아 있다.
품질 규칙은 전체 분포를 보고 최적화한 임계값이 아닌, 명시적인 탐색 규칙이다.

- QD 시각화/초기 기울기: `0 < QD ≤ 1.5 Ah`만 사용. EOL 이하를 삭제하는 필터가 아니다.
- 온도·IR: 0 이하를 결측 후보로 처리. 충전시간: 0 이하 또는 60분 초과는 중앙값 계산에서 제외.
- 초기 요약: 원래 사이클 번호 2~100. 첫 사이클은 Batch 간 초기화 차이 때문에 제외.
- Qdlin: 10·100사이클이 각각 1,000개 유한값인지 검증. 배열 위치를 사이클 번호로 착각하지 않는다.
- 이런 규칙의 민감도는 DAY 2 내부 검증에서 확인해야 한다.'''),
code('''display(quality.groupby(["batch", "field"])[["missing", "zero", "qd_outside_0_1_5", "charge_over60"]].sum())
display(features.groupby("batch")[["has_cycle5", "has_cycle100", "dq_valid"]].sum())'''),
md('''## 3. 정답 신뢰도와 분석 집합
전체 139개 셀을 원본 구조/곡선에 포함한다. 수명 분포는 유한한 정답 129개 기준이다.
상관관계와 모델 후보 분석은 다음 사유를 표시한 뒤 115개를 사용한다.

- Batch 1: `0~4`는 공개 코드상 다음 배치로 이어진 셀이고 현재 종료 QD도 높다. 필요한 2017-06-30 파일이 없으므로 수명 보정 대신 보류한다.
- Batch 1: `8, 10, 12, 13, 22`는 논문 공개 코드의 종료 미확인 셀이다. 현재 종료 QD도 0.91~0.97 Ah로 남아 있다.
- Batch 2: 정답 NaN 8개는 특수 프로토콜이며 지도학습/수명 상관분석에서 제외한다. 10/100사이클 배열 자체는 존재한다.
- Batch 3: 공개 코드의 품질 경고 셀 `2, 23, 32, 37, 42, 43`은 주분석에서 제외한다. 이 중 23, 32는 정답도 NaN이다.

이 제외는 '짧게 산 셀을 지우는 것'이 아니다. 실제로 짧은 유효 셀은 유지한다.
모든 셀을 사용한 상관관계도 함께 저장하여 제외에 따른 변화를 확인한다.'''),
code('''audit = features[["cell_id", "batch", "cycle_life", "last_qd", "dq_valid", "analysis_eligible", "review_reason"]]
display(audit[~audit.analysis_eligible])
print("주분석 셀 수")
display(features.groupby("batch").analysis_eligible.sum())'''),
md('''## 4. 재현성 점검
파일 전체 SHA-256과 라이브러리 버전을 저장한다. 데이터 파일명만 같고 내용이 달라지는 경우를 구분할 수 있다.
아래 검사는 실제 배열 크기, 사이클 일치, 셀 수, 피처 유한값을 확인한다.'''),
code('''import hashlib, json
fingerprints = []
for _, row in manifest.iterrows():
    path = ROOT / "data" / row["file"]
    sha = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
    fingerprints.append({"file": row["file"], "sha256": sha.hexdigest(), "bytes": path.stat().st_size})
pd.DataFrame(fingerprints).to_csv(OUT / "tables" / "data_fingerprints.csv", index=False)
display(pd.DataFrame(fingerprints))
assert len(cells) == 139 and features.cell_id.is_unique
assert summary.cells.tolist() == [46, 47, 46]
assert summary.missing_labels.tolist() == [0, 8, 2]
assert summary.eligible.tolist() == [36, 39, 40]
assert features.dq_valid.all()
assert np.isfinite(features.loc[features.analysis_eligible, "log_dq_var"]).all()
print("PASS: 139개 셀, 129개 유효 정답, 115개 주분석 셀, 모든 ΔQ 배열 검증 완료")
display(json.loads((OUT / "environment.json").read_text()))'''),
md('''## 확인 결과를 어떻게 이해하는 방법
Scratch의 38,811행은 **38,811개의 독립적인 배터리**가 아니다. Batch 1 배터리는 46개이며 주분석은 36개이다.
따라서 모델은 셀당 1행으로 학습해야 하며, 무작위로 사이클 행을 나누면 안 된다.
다음 `02_day1_eda.ipynb`에서 다섯 질문의 그래프·해석·모델 전략을 확인한다.

출처: 교수님 제공 Scratch 및 DAY 1 과제 화면, [논문](https://doi.org/10.1038/s41560-019-0356-8),
[저자 공개 로딩 코드](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation/blob/master/Load%20Data.ipynb).''')]

nb2=[md('''# 02. DAY 1 EDA와 모델 설계 전략
SKALA 3반 · U095 · 이연주 | 2026-10-01

**문제:** 초기 100사이클의 관측으로 배터리의 전체 사이클 수명을 예측할 수 있는가?
**목표:** Batch별 다섯 질문 분석 → 관찰 → 해석 → 피처와 모델 전략.

이 노트북만 Run All 해도 모든 분석 표·그림을 다시 만들 수 있다. 데이터 지문은 `01_data_check.ipynb`에서 생성한다.
분석용 함수는 `src/day1_analysis.py`, 보완 분석은 `src/day1_review.py`에 모았다. Scratch의 변수·출력·실행 상태를 참조하지 않으며 원본 MAT 파일에서 독립 계산한다. 아래 셀은 순서대로 실행하며, 모델 성능은 아직 계산하지 않는다.
데이터 범위와 품질 규칙은 `01_data_check.ipynb`와 같다.'''),code(setup),code(common),
md('''## Q1. Cycle Life 분포는 어떻게 생겼는가?
수명 NaN은 히스토그램의 분모에서 제외한다. `<500` 단수명, `500~1000` 중간수명, `>1000` 장수명이다.
과제의 150~2,300 축을 유지하되 현재 유효 정답의 실제 범위와 구분한다.'''),
code('''da.fig_distribution(features, OUT)
plt.show()
display(summary[["batch", "labels", "missing_labels", "min", "median", "mean", "max", "short_n", "short_pct", "long_n", "long_pct"]].round(2))'''),
md('''**관찰:** 중앙값은 Batch 1=858.5, Batch 2=472.0, Batch 3=1005.5이다. Batch 2는 유효 39개 중 28개(71.8%)가 500 미만이다. Batch 1·3에는 500 미만이 없다.

**해석:** 제공된 Batch는 동일한 수명 분포에서 무작위로 나뉜 집합이 아니다. 평균 성능 한 숫자로 일반화를 판단하기 어렵다. 150~2,300은 과제의 축 범위이지 현재 파일의 실제 최솟값·최댓값이 아니다(392~1,935).

**시사점:** 회귀 오차를 Batch별·수명 구간별로 보고한다. 품질 점검 후 Batch 1의 최대 수명이 1,074이므로 Batch 3 장수명은 외삽 문제이다.'''),
code('''shortest = features[features.analysis_eligible].sort_values("cycle_life").groupby("batch").head(1)
display(shortest[["cell_id", "cycle_life", "policy", "qd_slope", "log_dq_var", "tavg_median", "ir_median"]])
outliers = pd.read_csv(OUT / "tables/lifetime_outliers.csv")
display(outliers.groupby(["batch", "iqr_flag"]).size().rename("cells"))
display(outliers[outliers.iqr_flag.ne("within")])'''),
md('''**짧은 셀 점검:** b1c20(534), b2c19(392), b3c28(541)을 유지하고 초기 기울기·ΔQ·온도·정책을 함께 본다. b2c19는 초기 용량 기울기가 양수인데도 수명이 짧다. 초기 용량이 증가한다고 장수한다고 단정할 수 없다. 이 자료만으로 개별 셀의 고장 원인을 확정하지 않는다.

**통계적 이상치:** Batch별 Q1-1.5×IQR / Q3+1.5×IQR 기준에서 하위 이상치는 0개이다. 상위는 Batch 1=0, Batch 2=9, Batch 3=3개이다. Batch 2의 상위 9개는 모두 newstructure 조건이다. 이는 다른 하위 집단의 가능성을 보여주며 자동 삭제하지 않는다. '단수명(<500)'과 '통계적 이상치'는 서로 다른 기준이다.'''),
md('''**최단 셀의 상대적 위치:** 다음 표는 각 Batch 최단 셀의 초기 신호를 해당 Batch 중앙값 및 같은 정책의 다른 셀과 비교한다. 백분위는 해당 값 이하인 셀 비율이다. ΔQ 로그분산은 Batch 내 97.2/82.1/100백분위이다. b1c20·b2c19와 같은 정책의 다른 셀도 각각 559·408사이클로 짧아 충전 조건과의 관련 가능성이 있다. b3c28의 Tavg는 35.71°C로 Batch 중앙값 34.13°C보다 높다. 대조 실험이 아니므로 원인 증명이 아니라 열화 위험 신호를 좁히는 근거이다.'''),
code('''review.evidence_tables(features, OUT)
display(pd.read_csv(OUT / "tables/shortest_cell_comparison.csv").round(4))'''),
md('''## Q2. 방전용량은 어떻게 감소하는가?
전체 궤적은 EDA에 사용한다. 색은 실제 기록된 수명에 연결하며 NaN 정답은 회색이다.
EOL 참고선은 논문 공칭용량 1.1 Ah의 80%인 **0.88 Ah**이다. 제공 정답을 이 선으로 임의 재계산하지 않는다.'''),
code('''da.fig_degradation(cells, features, OUT)
plt.show()
da.fig_early(cells, features, OUT)
plt.show()
display(review.slope_sensitivity(cells, features, OUT).round(3))'''),
md('''**관찰:** 초기에는 궤적이 겹치며 일부 셀은 용량이 증가한다. 이후 많은 셀에서 감소가 가속된다. Batch 1에는 종료 시에도 높은 용량을 유지하는 셀이 있다.

**해석:** 전체 종료 시점을 모든 셀에서 동일하게 확인했다고 볼 수 없다. 0 또는 과도한 용량을 연결하면 선이 왜곡되므로 품질 규칙 밖 값은 NaN으로 끊었다. Batch 2 일부 초기 곡선에는 급락 후 복귀가 있다. 범위 안의 값을 자동 삭제하지 않고, 초기 2~100사이클에만 11점 이동 중앙값을 적용한 기울기와 원래 기울기의 상관을 비교했다. 순간 변화는 지속적인 열화와 구분해야 한다. 0.02 Ah 편차는 탐색용 표시 기준이며 고장 판정 기준이 아니다.

**시사점:** 초기 용량 단독보다 변화량·ΔQ 형태를 고려한다. 전체 곡선의 knee와 마지막 용량은 미래 정보이므로 입력 피처로 사용하지 않는다.'''),
code('''da.fig_representatives(cells, features, OUT)
plt.show()
rep = features[features.cell_id.isin(da.representatives(features))]
display(rep[["cell_id", "cycle_life", "knee_cycle", "slope_before", "slope_after", "knee_gain", "knee_accepted"]].round(5))'''),
md('''**Knee 방법:** 주분석 집합에서 Batch별 최단·중앙 순위·최장 셀을 선택한다. 11점 이동 중앙값에 연속된 두 직선을 맞춰 SSE가 가장 작아지는 지점을 찾는다. 단일 직선 대비 SSE 25% 이상 감소, 후기 기울기 더 음수, 후기 기울기 < -0.0001 Ah/cycle이면 후보로 표시한다.

점선은 알고리즘상 변화점 후보이다. 물리적 열화 기전이나 정확한 knee의 정답이 아니며 평활화·구간 선택에 민감하다. Batch 1·3의 최단 대표는 500 미만 셀이 아니므로 '단수명(<500)'으로 부르지 않는다.'''),
md('''## Q3. ΔQ(V)는 초기 사이클에서 차이를 보이는가?
동일한 전압축에서 `ΔQ = Q100 - Q10`을 계산한다. 리스트 번호 10을 곧 사이클 10으로 쓰지 않고 실제 summary.cycle로 위치를 찾는다.
분산은 표본분산(ddof=1), 왜도는 bias=False, 첨도는 초과첨도, 면적은 전압축을 오름차순으로 정렬해 적분한다.'''),
code('''# 핵심 식을 한 셀에서 직접 확인한다. 전체 계산은 같은 식을 적용한다.
example = cells[0]
delta_q = example["qcurves"][100] - example["qcurves"][10]
assert len(delta_q) == 1000
check = np.log10(np.var(delta_q, ddof=1))
assert np.isclose(check, features.loc[features.cell_id.eq(example["cell_id"]), "log_dq_var"].iloc[0])
print("예시 셀:", example["cell_id"], "log10 Var(ΔQ) =", check)
da.fig_delta(cells, features, OUT)
plt.show()
da.fig_delta_groups(cells, features, OUT)
plt.show()'''),
md('''**관찰:** ΔQ의 형태와 퍼짐이 셀마다 다르다. Batch 2는 단수명/장수명 집단을 직접 비교할 수 있지만, Batch 1·3은 단수명(<500) 집단이 없어 중간/장수명으로 비교한다. 음영은 집단의 IQR이며 신뢰구간이 아니다.

**해석:** 전압에 따른 용량 변화가 초기 총용량에 드러나지 않는 차이를 담는다. 다만 상관관계는 인과관계도 모델 성능도 아니다.

**시사점:** `log_dq_var`를 핵심 피처로 선정한다. ΔQ 평균·면적·표준편차를 무조건 모두 넣지 않고 중복을 점검한다.'''),
code('''da.fig_feature_scatter(features, OUT)
plt.show()
display(corr[corr.feature.eq("log_dq_var")].round(3))
da.supplementary_tables(features, corr, OUT)
display(pd.read_csv(OUT / "tables/delta_correlation_uncertainty.csv").round(3))'''),
md('''**안정성:** 품질 점검 후 ΔQ 로그분산의 Pearson r은 -0.827/-0.902/-0.742, Spearman ρ는 -0.826/-0.709/-0.759이다. 방향이 세 Batch에서 유지된다.
정책 단위 군집 부트스트랩 1,000회(난수 42) 구간도 저장했다. 정책 수가 적으므로 구간 역시 탐색적인 불확실성 표현이다.
모든 유효 정답을 사용한 결과와 품질 점검 후 결과는 `correlations.csv`에서 대조할 수 있다.'''),
md('''**Batch 내부 상대 비교 보완:** 절대 기준 <500이 없는 Batch 1·3도 수명 하위/상위 25% 곡선을 비교한다. 이는 과제의 단수명/장수명 정의를 바꾸는 것이 아니라 보충 분석이다. 세 Batch에서 하위 집단의 로그분산 중앙값이 더 높고, 곡선의 전압별 변화가 더 크다는 신호를 확인한다. 표본 수·컷오프·범위와 IQR을 함께 제시한다. 그룹 분리는 전체 수명 정답을 사용하는 EDA이며 모델 입력에는 사용하지 않는다.'''),
code('''fig, relative = review.relative_delta(cells, features, OUT)
plt.show()
display(relative.round(3))'''),
md('''## Q4. 충전 조건과 수명의 관계는?
프로토콜의 C1, 전환 SOC, C2를 분리한다. `(new)` 표시는 원문의 `-newstructure`이며 삭제하지 않고 구분한다.
단순한 C1 평균이 아니라 0~80% SOC 두 단계의 등가 C-rate를 사용한다:
`C_eq = 0.8 / [p/C1 + (0.8-p)/C2]`, p=전환 SOC/100. 휴지·CV·특수 프로토콜에는 동일한 해석을 강제하지 않는다.'''),
code('''da.fig_policy(features, OUT)
plt.show()
display(da.policy_table(features).round(2))
da.fig_rate(features, OUT)
plt.show()
display(pd.read_csv(OUT / "tables/rate_fade_correlations.csv").round(3))'''),
md('''**관찰:** Batch 1에서는 C_eq-수명 r=-0.580이지만 Batch 2는 +0.334, Batch 3는 -0.020이다. Batch 3의 C_eq 범위는 약 4.794~4.813으로 매우 좁다.
Batch 2의 동일한 `4.8C(80%)-4.8C`도 일반 프로토콜 평균 483.6(n=5), newstructure 평균 871.7(n=3)으로 다르다. 구조/실험 조건이 함께 달라질 수 있다.

**해석:** '빠른 충전이면 언제나 단수명'이라는 전역 결론을 낼 수 없다. 표준편차 막대는 평균의 신뢰구간이 아니며 n=1이면 표준편차를 추정할 수 없다.
실측 전류는 사이클 10 양의 전류의 95백분위수로 요약했고, 초기 QD 기울기와의 관계도 Batch별로 달랐다.

**시사점:** 정책을 검증 그룹으로 사용하고 C_eq는 보조 피처 후보로 비교한다. newstructure의 정확한 물리적 의미를 자료만으로 지어내지 않는다.'''),
md('''**실측 충전 패턴 확인:** cycle 10에서 각 Batch의 최단·중앙 순위·최장 셀 전류를 직접 비교한다. 첫 양의 충전 전류부터 첫 음의 방전 전류 직전까지 표시한다. 패턴은 단계 전환과 후반 전류 감소를 보여준다. P95는 양의 전류 샘플의 크기 요약으로, 시간 가중 평균이나 충전 에너지·충전율 전체를 대신하지 않는다. 전류-초기 열화 상관은 위 표에서 확인하며, 전류 패턴만으로 개별 수명 차이의 원인을 확정하지 않는다.'''),
code('''fig, current = review.current_patterns(cells, features, OUT)
plt.show()
display(current.round(3))'''),
md('''## Q5. 어떤 신호가 수명과 연관되는가?
Pearson은 선형 관계, Spearman은 순위 관계를 본다. 전체 pooled 값과 Batch별 값을 함께 확인해야 Batch 차이를 피처 효과로 오인하지 않는다.'''),
code('''da.fig_correlations(corr, OUT)
plt.show()
fig, feature_pairs = review.collinearity(features, OUT)
plt.show()'''),
md('''**관찰:** ΔQ 로그분산은 안정적인 음의 연관성을 보인다. 충전시간은 +0.571/-0.919/+0.599로 방향이 바뀐다. 초기 QD 기울기도 +0.508/-0.445/+0.331이다.
Batch 1에서 Tavg-Tmax r=0.949, log_dq_var-dq_std r=0.991, ΔQ 평균-면적 r≈1이다. log10 Var=2log10 Std이므로 분산과 표준편차는 독립 정보가 아니다.

**해석:** 높은 pooled 상관계수 하나로 피처를 선택하면 Batch 교란과 중복을 놓친다. 온도·저항은 단독 상관이 약하다고 모든 정보를 배제할 수는 없으나, 작은 표본에서 우선순위를 낮춘다.

**시사점:** 소수 피처와 규제가 필요하다. ΔQ 파생 피처는 대표 하나를 우선 사용하고, 보조 피처는 학습 Batch 내부 교차검증의 추가 효과로 결정한다.'''),
md('''**핵심 피처 분포와 최강 관계 점검:** 박스는 중앙값·IQR, 점은 셀이다. 전처리 전의 실제 분포를 비교한다. 배치별 범위 차이를 확인한 후 대치·표준화는 반드시 학습 fold에서만 적합한다. 아래 표는 각 Batch의 절대 Pearson/Spearman 상관 상위 3개를 각각 보여준다. 최고 순위와 여러 Batch에 걸친 안정성을 구분하며, 안정성과 단순성을 이유로 log_dq_var를 주피처로 선택한다.'''),
code('''fig, distribution = review.feature_distributions(features, OUT)
plt.show()
display(distribution.round(5))
for method in ["pearson", "spearman"]:
    ranked = corr[corr.scope.eq("quality_screened") & corr.batch.ne("Pooled")].copy()
    ranked["abs_r"] = ranked[method].abs()
    display(ranked.sort_values("abs_r", ascending=False).groupby("batch").head(3)[["batch", "feature", "n", method]])
display(feature_pairs[feature_pairs.abs_r_ge_0_9].sort_values(["batch", "pearson"], ascending=[True, False]).round(3))
display(pd.read_csv(OUT / "tables/screening_sensitivity.csv").round(3))'''),
md('''## 모델 설계 전략: 초기 100사이클 회귀
**타깃:** 제공 `cycle_life`(사이클). 학습에서는 `log10(cycle_life)` 변환을 비교하고 예측을 원래 단위로 복원해 평가한다.
500 기준 분류는 Batch 1이 0:46(주분석 0:36)로 불균형이 심하다. 단수명 표본이 없어 안정적인 분류 검증이 어렵고 수치 예측의 운영 활용도도 높아 회귀를 선택한다.

**피처 계획**

|집합|피처|EDA 근거|
|---|---|---|
|F1 주안|log_dq_var|세 Batch에서 방향 일치, 한 개 대표 변수|
|F2 비교|F1 + qd_slope + c_equiv|초기 변화와 정책 보완, Batch 1 내부 검증으로 추가 가치 판정|
|F3 민감도|F2 + tavg_median + ir_median|온도/저항의 조합 효과만 제한적으로 확인|
|우선 제외|dq_std, dq_mean, dq_area, Tmax 동시 투입|다중공선성/중복|
|보류|충전시간, 정책 문자열 원핫, newstructure|Batch별 부호 변화/희소 범주/학습 Batch에 없는 조건|
|입력 금지|cell_id, batch, cycle_life 파생값, n_cycles, last_qd, knee|식별자·정답·미래 정보|

**후보 모델을 EDA와 연결**

1. DummyRegressor 중앙값: 실제로 배웠는지 확인할 기준선.
2. Ridge / ElasticNet: 표본이 작고 ΔQ 파생량이 중복되므로 규제 선형 모델 우선.
3. 얕은 RandomForest: 비선형 비교군. max_depth 2~3, min_samples_leaf 3~5; 학습 범위 밖 수명 외삽에 한계.

모델 적합과 성능 비교는 DAY 2에 수행한다. DAY 1 산출물에 미측정 성능을 적지 않는다.'''),
code('''split = pd.read_csv(OUT / "tables/planned_split.csv")
display(split.groupby("planned_split").agg(cells=("cell_id", "size"), policies=("policy", "nunique")))
display(split)
assert not set(split[split.planned_split.eq("train")].policy) & set(split[split.planned_split.eq("validation")].policy)
print("PASS: 정책 그룹과 셀 ID가 train/validation에 겹치지 않는다.")'''),
md('''## DAY 2 실행 설계와 평가의 한계
- Batch 1 주분석 36개를 정책 단위 GroupShuffleSplit(test_size=0.2, random_state=42) → train 29개/16정책, validation 7개/4정책.
- Train 안에서 4-fold GroupKFold. 결측 대치·StandardScaler·피처 선택을 각 fold의 train에만 fit하는 Pipeline 사용.
- 후보: Ridge alpha [0.1,1,10,100], ElasticNet alpha [0.001,0.01,0.1]와 l1_ratio [0.2,0.8]; 좁은 탐색으로 과도한 튜닝 방지.
- 모델/피처 집합을 train CV로 고른 뒤 validation을 한 번 확인. 절차 동결 후 Batch 1 36개로 재학습, Batch 2 39개, Batch 3 40개에서 평가.
- 주지표 MAPE(%), 보조 MAE/RMSE(사이클), R². 수명 구간별 잔차 및 큰 오차 셀을 공개.
- DAY 1 요구사항 때문에 Batch 2·3의 정답 분포/상관을 이미 관찰했다. 따라서 향후 평가는 **EDA에서 관찰한 외부 Batch 평가**이며 완전한 블라인드 테스트가 아니다. 새 미관찰 배치로 후속 검증해야 한다.
- Batch 1 내부 validation 역시 전체 Batch 1 EDA 후 설계됐다는 한계를 명시한다.
- 논문 9.1%는 다른 정제/분할 조건의 참고 성능이다. 직접 재현 성능이나 달성 보장으로 쓰지 않는다.

ESS 활용은 셀 수명 선별·교체 계획의 연구 단계이다. 실제 ESS의 온도, 부하, 달력 열화, 팩 불균형 조건은 추가 검증이 필요하다.
운영 시 피처 분포 변화(Data Drift), 정답 확보 후 오차 증가(Model Drift)를 모니터링하고 새 Batch 검증 후 재학습한다.'''),
code('''# 산출물 누락 검증 (학습은 하지 않음)
expected = ["01_life_distribution", "02_degradation", "03_representatives_knee", "04_delta_curves", "05_delta_groups", "06_feature_scatter", "07_policy_means", "08_rate_and_fade", "09_correlations", "10_collinearity", "11_early_cycles", "12_feature_distributions", "13_delta_relative_groups", "14_current_patterns"]
for name in expected:
    assert (OUT / "figures" / f"{name}.png").stat().st_size > 1000
assert len(features[features.analysis_eligible]) == 115
print("PASS: 필수 그래프 11개 + PDF용 정책 그래프 3개 + 보완 그래프 3개, 표, 분리 계획 생성 완료")'''),
md('''## 참고자료와 다음 읽기 순서
1. 교수님 제공 `30-ESSHealth-scratch.ipynb`, DAY 1 과제 화면 3장, 저장된 강의 webarchive.
2. Severson et al. (2019), Nature Energy 4, 383–391. https://doi.org/10.1038/s41560-019-0356-8
3. 저자 공개 코드: https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation

먼저 Q1의 Batch 차이 → Q3의 ΔQ 핵심 신호 → Q5의 피처 중복 → 모델 설계 순서로 이해하면 전체 흐름을 잡기 쉽다.
분석/문서 초안에 AI 도구를 활용했으며 제출자는 수치와 해석을 확인하고 설명할 수 있어야 한다.''')]

for name,cells in [('01_data_check.ipynb',nb1),('02_day1_eda.ipynb',nb2)]:
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3 (.venv)','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11'}})
    (ROOT/'notebooks').mkdir(exist_ok=True)
    nbf.write(nb,ROOT/'notebooks'/name)
    print(name,'cells',len(cells))
