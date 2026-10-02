"""Record tested package versions and artifact inventory after PDF QA."""
from pathlib import Path
import importlib.metadata as im,json,subprocess,sys,hashlib
ROOT=Path(__file__).resolve().parents[1]
PY=ROOT/'.venv/bin/python'
packages=['numpy','pandas','scipy','mat73','h5py','matplotlib','seaborn','scikit-learn','jupyter','nbformat','nbclient','nbconvert','ipykernel']
code='import importlib.metadata as m,json;print(json.dumps({p:m.version(p) for p in '+repr(packages)+'}))'
versions=json.loads(subprocess.check_output([str(PY),'-c',code],text=True))
for p in ['reportlab','pypdf','Pillow']:versions[p]=im.version(p)
content='# 검증에 사용한 주요 패키지 버전. 전체 전이 의존성 잠금은 아니다.\n'
content+='\n'.join(f'{k}=={v}' for k,v in sorted(versions.items()))+'\n'
(ROOT/'requirements-day1.lock.txt').write_text(content)
env=json.loads((ROOT/'results/day1/environment.json').read_text())
env['report_python']=sys.version.split()[0]
env['report_libraries']={p:im.version(p) for p in ['reportlab','pypdf','Pillow']}
env['execution_note']='분석: 프로젝트 .venv Python 3.11 / PDF: Codex bundled Python 3.12. 주요 패키지 버전을 기록했으며 단일 신규 환경 설치는 별도 검증하지 않음.'
(ROOT/'results/day1/environment.json').write_text(json.dumps(env,ensure_ascii=False,indent=2))
paths=[*sorted((ROOT/'notebooks').glob('0[12]*.ipynb')),*sorted((ROOT/'results/day1/figures').glob('*.png')),*sorted((ROOT/'results/day1/tables').glob('*.csv')),ROOT/'results/DS-MINI-Design-SKALA_3반-U095_이연주.pdf']
paths += [*sorted((ROOT/'src').glob('*.py')),*sorted((ROOT/'scripts').glob('*.py')),ROOT/'README.md',
          ROOT/'requirements-day1.txt',ROOT/'requirements-day1.lock.txt']
paths += [ROOT/'results/day1'/name for name in ['environment.json','verification.json','notebook_execution.json','submission_review.json']]
inventory=[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]
(ROOT/'results/day1/artifact_manifest.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2))
print('Recorded',len(inventory),'artifacts')
