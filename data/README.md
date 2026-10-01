# Data

대용량 원본 데이터는 Git 저장소에 포함하지 않습니다.

출처: [MIT–Stanford Battery Dataset](https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle)

## 필요한 파일

| 파일 | 역할 | 필수 여부 |
|---|---|---|
| `2017-05-12_batchdata_updated_struct_errorcorrect.mat` | Batch 1 · 학습 및 내부 검증 | 필수 |
| `2018-02-20_batchdata_updated_struct_errorcorrect.mat` | Batch 2 · 최종 테스트 | 필수 |
| `2018-04-12_batchdata_updated_struct_errorcorrect.mat` | Batch 3 · 추가 일반화 평가 | 선택 |

다운로드한 `.mat` 파일은 이 폴더에 두되 Git에는 커밋하지 않습니다.

