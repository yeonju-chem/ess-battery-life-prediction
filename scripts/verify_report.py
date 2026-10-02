"""Check final PDF text, notebook execution and the manual rendering QA record."""
from pathlib import Path
import json,re,hashlib
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/day1'
pdf=ROOT/'results/DS-MINI-Design-SKALA_3반-U095_이연주.pdf'
reader=PdfReader(pdf);texts=[p.extract_text() for p in reader.pages]
assert len(texts)==27, 'Unexpected report pagination; re-render and inspect.'
assert all(len(s)>250 for s in texts), 'Possible overflow or blank page.'
whole='\n'.join(texts)
assert not re.search(r'[가-힣]+(?:습니다|합니다|입니다|됩니다|까요|세요)',whole)
assert 'SKALA 3반' in whole and 'U095 이연주' in whole
for phrase in ['Q1','Q2','Q3','Q4','Q5','Pearson','Spearman','500','1,000','Ridge','ElasticNet','MAPE','블라인드']:
    assert phrase in whole, phrase
assert 'b1c20 / b1c11 / b1c5' in texts[8]
assert all('DAY 1' in s for s in texts)
integrity=json.loads((OUT/'verification.json').read_text())
assert all(x['passed'] for x in integrity['checks'])
execution=json.loads((OUT/'notebook_execution.json').read_text())
assert all(x['errors']==0 and x['all_executed'] for x in execution)
review={'pdf_pages':len(texts),'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),
        'executed_code_cells':sum(x['code_cells'] for x in execution),
        'integrity_checks_passed':len(integrity['checks']),
        'figures':len(list((OUT/'figures').glob('*.png'))),
        'csv_tables':len(list((OUT/'tables').glob('*.csv'))),
        'narrative_style':'declarative Korean',
        'rubric_coverage':{'EDA_50':'Q1-Q5, per-batch distributions and statistical analysis',
                           'EDA_strategy_30':'observations, interpretation, implications and limitations',
                           'model_strategy_20':'regression target, features, candidates and validation'},
        'note':'Coverage check, not a predicted grade. Rendering must be reviewed separately.'}
(OUT/'submission_review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2))
print(json.dumps(review,ensure_ascii=False,indent=2))
