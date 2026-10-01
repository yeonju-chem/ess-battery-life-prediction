# ESS 배터리 수명 예측 Mini Project

MIT–Stanford Battery Dataset의 초기 충방전 사이클을 이용해 배터리 수명을 분석하고 예측하는 프로젝트입니다.

## 현재 단계

- [x] 과제 요구사항 확인
- [x] 원본 Scratch notebook 보존
- [x] 프로젝트 폴더 구성
- [ ] Batch 데이터 확보
- [ ] 데이터 구조 및 품질 확인
- [ ] Batch별 EDA
- [ ] 회귀 또는 분류 전략 확정
- [ ] 모델 개발 및 평가
- [ ] 결과·한계·ESS 도메인 해석

## 폴더 구조

```text
ess-battery-project/
├── data/
│   └── README.md
├── notebooks/
│   └── 00_scratch_reference.ipynb
├── src/
├── results/
├── requirements.txt
└── README.md
```

## 평가 원칙

- Batch 1: 학습 및 내부 검증
- Batch 2: 필수 최종 테스트. 모델 선택에 사용하지 않음
- Batch 3: 추가 일반화 평가용
- 모든 분리는 배터리 셀 단위로 수행
- 전처리는 train 데이터에만 `fit`
- 무작위 단계는 `random_state=42`로 고정

## 참고문헌

Severson et al. (2019). *Data-driven prediction of battery cycle life before capacity degradation*. Nature Energy, 4, 383–391.

