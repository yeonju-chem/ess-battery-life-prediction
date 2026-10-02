"""Rebuild the DAY 1 Korean PDF from verified analysis tables and figures.
Usage: python scripts/build_report.py (requires reportlab, Pillow).
"""
from pathlib import Path
import csv, json, math, sys
from xml.sax.saxutils import escape
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image as PILImage

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/day1'
PDF=ROOT/'results/DS-MINI-Design-SKALA_3반-U095_이연주.pdf'
FONT=Path('/System/Library/Fonts/Supplemental/AppleGothic.ttf')
if not FONT.exists():
    import os
    FONT=Path(os.environ.get('ESS_KOREAN_FONT','assets/NanumGothic.ttf'))
pdfmetrics.registerFont(TTFont('KR',str(FONT)))
NAVY=colors.HexColor('#142F38');BLUE=colors.HexColor('#237A86');GREEN=colors.HexColor('#127D69');GRAY=colors.HexColor('#607278');LIGHT=colors.HexColor('#F1F5F5')
W,H=landscape(A4);CW=W-88
styles={
 'title':ParagraphStyle('title',fontName='KR',fontSize=29,leading=38,textColor=NAVY,spaceAfter=18,wordWrap='CJK'),
 'h':ParagraphStyle('h',fontName='KR',fontSize=21,leading=28,textColor=NAVY,spaceAfter=15,wordWrap='CJK'),
 'body':ParagraphStyle('body',fontName='KR',fontSize=10.5,leading=15.5,textColor=colors.HexColor('#243447'),spaceAfter=8,wordWrap='CJK'),
 'small':ParagraphStyle('small',fontName='KR',fontSize=9,leading=13,textColor=GRAY,spaceAfter=6,wordWrap='CJK'),
 'cell':ParagraphStyle('cell',fontName='KR',fontSize=10,leading=14,textColor=colors.HexColor('#243447'),wordWrap='CJK'),
 'head':ParagraphStyle('head',fontName='KR',fontSize=10,leading=14,textColor=colors.white,wordWrap='CJK')}
story=[]
def p(s,kind='body'):return Paragraph(s,styles[kind])
def txt(s,kind='body'):
 if s.startswith(('시사점:', '해석·시사점:', '관찰·시사점:', '선택:', '평가의 한계:')):
  box=Table([[p(s,kind)]],colWidths=[CW])
  box.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#EAF4F0')),
      ('LINEBEFORE',(0,0),(0,0),3,GREEN),('LEFTPADDING',(0,0),(-1,-1),11),
      ('RIGHTPADDING',(0,0),(-1,-1),10),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
  story.extend([box,Spacer(1,8)])
 else:story.append(p(s,kind))
def page(title):
 if story:story.append(PageBreak())
 story.append(p(title,'h'))
def fig(name,width=CW,maxheight=300):
 path=OUT/'figures'/f'{name}.png'; im=PILImage.open(path);w,h=im.size
 scale=min(width/w,maxheight/h);story.append(Image(str(path),width=w*scale,height=h*scale));story.append(Spacer(1,7))
def table(data,widths=None,padding=7):
 arr=[[p(escape(str(v)), 'head' if i==0 else 'cell') for v in row] for i,row in enumerate(data)]
 t=Table(arr,colWidths=widths or [CW/len(data[0])]*len(data[0]),repeatRows=1,hAlign='LEFT')
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),padding),('BOTTOMPADDING',(0,0),(-1,-1),padding),('LEFTPADDING',(0,0),(-1,-1),9),('RIGHTPADDING',(0,0),(-1,-1),9),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,LIGHT]),('LINEBELOW',(0,0),(-1,0),.5,NAVY),('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#D4DFE8'))]))
 story.append(t);story.append(Spacer(1,10))
def read(name):
 with (OUT/'tables'/f'{name}.csv').open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
S=read('batch_summary');T=read('cell_features');C=read('correlations');Q=read('quality_by_cell_field');CI=read('delta_correlation_uncertainty');R=read('rate_fade_correlations')
def n(v,d=1):
 try:return f'{float(v):,.{d}f}'
 except:return 'NA'
def r(feat,b,method='pearson',scope='quality_screened'):
 return next(float(x[method]) for x in C if x['feature']==feat and x['batch']==b and x['scope']==scope)

hero_title=ParagraphStyle('hero_title',parent=styles['title'],textColor=colors.white,fontSize=30,leading=38,spaceAfter=8)
hero_sub=ParagraphStyle('hero_sub',parent=styles['body'],textColor=colors.HexColor('#C0DBD7'),fontSize=11,leading=17)
hero=Table([[Paragraph('ESS 배터리 수명 예측',hero_title)],
            [Paragraph('DAY 1  /  EDA 기반 모델 설계 전략',hero_sub)],
            [Paragraph('SKALA 3반  ·  U095 이연주  ·  2026.10.01',hero_sub)]],colWidths=[CW])
hero.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),NAVY),('LEFTPADDING',(0,0),(-1,-1),22),
                         ('RIGHTPADDING',(0,0),(-1,-1),22),('TOPPADDING',(0,0),(-1,0),20),
                         ('BOTTOMPADDING',(0,-1),(-1,-1),16)]))
story.extend([hero,Spacer(1,16)])
metric_style=ParagraphStyle('metric',parent=styles['title'],fontSize=23,leading=27,textColor=GREEN,spaceAfter=3)
cards=Table([[Paragraph('139',metric_style),Paragraph('129',metric_style),Paragraph('115',metric_style)],
             [p('원본 배터리 셀','small'),p('유효 수명 정답','small'),p('품질 점검 후 주분석 셀','small')]],colWidths=[CW/3]*3)
cards.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),LIGHT),('LEFTPADDING',(0,0),(-1,-1),16),
                          ('TOPPADDING',(0,0),(-1,0),10),('BOTTOMPADDING',(0,-1),(-1,-1),8)]))
story.extend([cards,Spacer(1,12)])
txt('초기 100사이클의 방전 곡선 변화로 수명을 예측하는 회귀 전략을 설계했다. 제공된 세 Batch를 직접 분석하고, 데이터 품질·배치 차이·피처 중복을 검증 설계에 반영했다.')
table([['분석 범위','확인 결과','전략으로 연결'],
 ['139개 원본 셀 / 3개 Batch','유효 수명 129개, 주분석 115개','정답 결측·종료 미확인·품질 경고를 기록'],
 ['핵심 신호: log10 Var(ΔQ)','Batch별 Pearson r: -0.827 / -0.902 / -0.742','한 개 대표 피처부터 규제 회귀 시작'],
 ['수명 분포 이동','Batch 2 단수명 71.8%, Batch 3 장수명 52.3%','Batch별 오차와 외삽 한계 보고'],
 ['분류 타깃 500 기준','Batch 1: 0 대 46, 주분석 0 대 36','수치 수명 회귀 선택']], [160,285,CW-445])
txt('읽는 순서: 데이터 신뢰도 → 다섯 질문의 관찰·해석 → 피처 선정 → 모델/검증 설계. 모든 그림은 제공된 원본 파일로 재계산했다. 이 보고서는 DAY 1 전략 산출물이며 학습 성능은 아직 측정하지 않았다.','small')

page('01  데이터 범위와 구조를 먼저 확인했다')
table([['구분','파일 날짜','원본 셀','유효 정답','총 사이클 행','정책 수'],*[[x['batch'],['2017-05-12','2018-02-20','2018-04-12'][i],x['cells'],x['labels'],n(x['cycles'],0),x['policies']] for i,x in enumerate(S)]],[90,150,90,100,150,CW-580])
txt('구조: Batch → 배터리 셀 → summary(사이클별 요약) / cycles(사이클 내부 시계열). 셀당 수명·정책·전압축이 있으며, 분석용 ID는 b1c0처럼 Batch와 파일 내 인덱스로 구성했다.')
table([['확인 항목','검사 결과','처리 원칙'],
 ['초기 정보','139개 모두 사이클 5·100 및 Qdlin 10·100 배열 존재','summary.cycle로 실제 사이클 번호를 찾음'],
 ['Qdlin / Vdlin','각각 1,000점, 전압 2.0~3.5 V','동일 전압축에서 Q100-Q10, 유한값 검사'],
 ['결측','summary 수치의 NaN 0개, 수명 NaN 10개','0값은 결측 검사만으로 찾을 수 없어 별도 집계'],
 ['표본 단위','원본 summary 총 116,722행 / 실제 셀 139개','모델은 배터리 셀당 1행']], [150,285,CW-435])
txt('논문 공개 코드의 두 번째 파일은 2017-06-30이다. 이번 제공 파일은 2018-02-20(47셀)로 다르므로 논문의 124셀 실험을 그대로 재현한 것으로 표현하지 않는다. [1, 2]','small')

page('02  정답 신뢰도를 확인한 뒤 주분석 집합을 만들었다')
table([['Batch','원본 → 주분석','보류 / 제외 사유'],
 ['Batch 1','46 → 36','b1c0~4: 공개 코드상 계속 측정된 셀이나 후속 파일 부재. b1c8,10,12,13,22: 종료 미확인. 현재 종료 QD도 모두 0.91 Ah 초과.'],
 ['Batch 2','47 → 39','b2c22,23,35~40: cycle_life가 NaN. 특수 프로토콜 8개로, 정답을 임의 대치하지 않음.'],
 ['Batch 3','46 → 40','b3c2,23,32,37,42,43: 저자 공개 코드의 품질 경고 채널. 23,32는 정답도 NaN.']], [100,120,CW-220])
txt('원본 전체 분포·곡선은 보존하고, 수명 상관분석은 전체 유효 정답과 주분석 집합을 모두 계산했다. 짧은 수명 자체는 삭제 사유가 아니다. 제외 셀과 이유는 cell_audit.csv에서 확인할 수 있다.')
table([['탐색 규칙','이유 / 영향'],
 ['QD: 0 < QD ≤ 1.5 Ah','시각화·초기 기울기에서만 적용. Batch 1/2/3에서 48/11/0행이 범위 밖. 0.88 Ah 이하라는 이유로 지우지 않음.'],
 ['초기 요약: 사이클 2~100','첫 사이클 초기화 차이로 1번을 공통 제외. 온도·IR의 0 이하는 무효 후보, 충전시간 60분 초과는 중앙값에서 제외.'],
 ['정제는 확정 진실이 아닌 분석 규칙','주분석/전체 유효 정답의 민감도 결과를 함께 저장. 미래 종료 정보는 타깃 품질 검사용이며 입력 피처에 사용하지 않음.']], [225,CW-225])
txt('출처 [2]의 셀 경고 목록과 현재 원시 데이터 점검을 결합했다. Batch 1 수명 보정이나 Batch 간 셀 병합은 수행하지 않았다.','small')

page('Q1  Batch마다 수명 분포가 크게 다르다')
fig('01_life_distribution',maxheight=240)
table([['구분','평균 / 중앙값','범위','단수명 <500','장수명 >1000'],*[[x['batch'],f"{n(x['mean'])} / {n(x['median'])}",f"{n(x['min'],0)}~{n(x['max'],0)}",f"{x['short_n']}/{x['labels']} ({n(x['short_pct'])}%)",f"{x['long_n']}/{x['labels']} ({n(x['long_pct'])}%)"] for x in S]],[90,180,130,175,CW-575])
txt('관찰: Batch 2는 짧은 수명에, Batch 3은 긴 수명에 상대적으로 집중된다. 비율의 분모는 정답이 있는 셀이다. 과제 축은 150~2,300으로 유지했고 실제 정답 범위는 392~1,935이다.')
txt('해석·시사점: 모든 Batch를 섞은 평균 성능만으로 일반화를 판단하지 않는다. Batch별·수명 구간별 오차를 보고하고, 학습 범위 밖 수명을 별도 분석한다.')

page('Q1  짧은 셀을 지우지 않고 원인을 탐색했다')
short=[]
for b in ['Batch 1','Batch 2','Batch 3']:
 g=[x for x in T if x['batch']==b and x['analysis_eligible']=='True'];short.append(min(g,key=lambda x:float(x['cycle_life'])))
table([['셀','수명','충전 정책','초기 QD 기울기','log10 Var(ΔQ)','Tavg 중앙값'],*[[x['cell_id'],n(x['cycle_life'],0),x['policy'],f"{float(x['qd_slope']):.6f}",n(x['log_dq_var'],3),n(x['tavg_median'],2)] for x in short]],[65,60,260,130,130,CW-645])
txt('관찰: b2c19는 초기 QD 기울기가 양수인데도 수명이 392로 짧다. b1c20과 b3c28은 초기 기울기가 음수이다. 같은 방향의 초기 용량 변화가 모든 Batch에서 같은 의미를 갖지는 않는다.')
txt('해석: 충전 정책, 온도, 저항, ΔQ를 함께 봐야 한다. 이 자료만으로 셀별 고장 기전이나 “왜 유독 짧은가”의 단일 원인을 확정할 수 없다. 상관 분석은 원인 후보를 좁히는 단계이다.')
txt('시사점: b1c20, b2c19, b3c28을 유지하고 DAY 2의 큰 오차 셀 분석 대상으로 추적한다. Batch 1·3의 최단 셀도 500 이상이므로 과제의 단수명 집단(<500)과 구분한다.')
txt('통계적 이상치: Batch별 1.5×IQR 기준에서 하위 이상치는 모두 0개, 상위는 0/9/3개이다. Batch 2 상위 9개는 모두 newstructure 조건이다. 짧은 수명과 통계적 이상치를 구분하며, 다른 실험 조건일 수 있어 자동 삭제하지 않는다. 셀별 목록은 lifetime_outliers.csv에 있다.','small')
table([['선택지','데이터 근거','결정'],['500 기준 이진 분류','Batch 1 전체 0:46 / 주분석 0:36. 단수명 셀이 없어 안정적 분류 검증이 어려움.','이번 과제에서 선택하지 않음'],['초기 100사이클 회귀','셀 간 수치 수명 차이와 ΔQ 신호를 활용. 운영에서 상대적 교체 우선순위 검토 가능.','cycle_life 회귀 선택']],[150,430,CW-580])

page('Q1  최단 셀의 신호를 같은 Batch와 비교한다')
short_evidence=read('shortest_cell_comparison')
table([['셀 / 수명','log ΔQ 분산: 셀 / Batch 중앙값','Batch 내 백분위','동일 정책 다른 셀: n / 수명 중앙값'],
 *[[x['cell_id']+' / '+n(x['cycle_life'],0),n(x['value'],3)+' / '+n(x['batch_median'],3),n(x['percentile_le'])+'%',x['other_same_policy_n']+' / '+n(x['other_same_policy_life_median'],1)]
   for x in short_evidence if x['feature']=='log_dq_var']],[115,240,125,CW-480])
txt('관찰: 최단 셀의 ΔQ 로그분산은 Batch 내 97.2 / 82.1 / 100백분위이다. b1c20·b2c19와 같은 정책의 다른 셀도 각각 559·408사이클로 짧아 충전 조건과의 관련 가능성이 있다. b3c28의 Tavg는 35.71°C로 Batch 중앙값 34.13°C보다 높다. 다만 대조 실험이 아니므로 단일 원인으로 확정하지 않는다.')
table([['가설','이 자료에서 확인할 수 있는 근거','판단'],
 ['초기 총용량 감소가 빠르기 때문인가?','b2c19는 기울기가 양수인데도 최단 수명이다.','모든 Batch에 통하는 단일 설명은 아니다.'],
 ['초기 전압별 용량 변화가 큰가?','위 상대 비교와 Q3의 ΔQ-수명 음의 관계를 함께 확인한다.','유력한 위험 신호이며 피처로 반영한다.'],
 ['온도·저항·충전 조건 때문인가?','셀별 값과 Batch 중앙값은 shortest_cell_comparison.csv에 기록한다.','관측 자료라 조건 교란을 제거하지 못한다.']],[205,320,CW-525])
txt('시사점: 최단 셀은 오류로 자동 삭제하지 않는다. 개별 고장 원인은 확정하지 않되, 상대적으로 큰 ΔQ 변화라는 관측 근거를 주피처 선택에 연결한다. DAY 2에서는 이 셀들의 예측 잔차를 추적한다.')

page('Q2  방전용량 감소는 일정한 직선이 아니다')
fig('02_degradation',maxheight=285)
txt('관찰: 다수의 셀에서 초기 용량이 약 1.1 Ah 부근에 모여 있다가 후반에 감소가 빨라진다. Batch 1에는 측정 종료 시 용량이 높게 남은 곡선이 있다. 정답 NaN은 회색, 나머지는 기록 수명에 따라 색을 매겼다.')
txt('해석: 전체 수명 종료가 모든 셀에서 같은 방식으로 관찰된 것은 아니다. 참고선 0.88 Ah는 공칭 1.1 Ah의 80%이다. 원래 cycle_life 필드와 임계값 교차 시점이 완전히 같다고 가정하지 않는다.')
txt('시사점: 기록 종료 길이를 수명 피처로 사용하지 않는다. 마지막 QD·전체 궤적·knee는 EDA 해석용이며 초기 예측 모델 입력에서 제외한다.')

page('Q2  초기 100사이클은 겹침과 초기 증가가 나타난다')
fig('11_early_cycles',maxheight=230)
txt('관찰: 주분석 집합에서 초기 곡선들은 좁은 용량 구간에 모이고 일부는 증가한다. 같은 초기 용량을 가진 셀의 전체 수명은 크게 다를 수 있다.')
ss=read('early_slope_sensitivity')
table([['Batch','원래 기울기-수명 r','평활 기울기-수명 r','순간 편차 셀 수'],
       *[[x['batch'],n(x['raw_slope_life_r'],3),n(x['smoothed_slope_life_r'],3),x['cells_with_local_deviation']] for x in ss]],[100,215,220,CW-535])
txt('Batch 2의 급락 후 복귀를 지속 열화로 단정하지 않는다. 초기 2~100사이클 안에서 11점 이동 중앙값을 적용해 민감도를 점검했다. 순간 편차는 원래 값과 이동 중앙값의 차이가 0.02 Ah를 넘는 경우이며, 고장 판정 기준이 아니다.','small')
txt('시사점: QD 기울기는 보조 후보로 남기되 단독 결정 신호로 삼지 않는다. ΔQ의 전압별 형태처럼 총용량이 놓치는 정보를 추가한다. 기울기는 원래 사이클 2~100의 유효 QD에 선형 적합한 값이다.')

page('Q2  Knee point는 관측 기반 후보로 표시했다')
fig('03_representatives_knee',maxheight=250)
rep=[]
for b in ['Batch 1','Batch 2','Batch 3']:
 g=sorted([x for x in T if x['batch']==b and x['analysis_eligible']=='True'],key=lambda x:(float(x['cycle_life']),x['cell_id']))
 rep.extend([g[0],g[len(g)//2],g[-1]])
table([['Batch','대표 셀 (최단 / 중앙순위 / 최장)','수명','knee 후보'],*[[b,' / '.join(x['cell_id'] for x in rep if x['batch']==b),' / '.join(n(x['cycle_life'],0) for x in rep if x['batch']==b),' / '.join(n(x['knee_cycle'],0) for x in rep if x['batch']==b)] for b in ['Batch 1','Batch 2','Batch 3']]],[95,315,175,CW-585])
txt('방법: 11점 이동 중앙값에 연속 두 직선을 적합했다. 단일 직선보다 SSE가 25% 이상 줄고 후기 기울기가 더 가파른 음수일 때 점선으로 표시했다. 상세 기울기·SSE 감소율은 노트북에 있다.')
txt('해석·한계: 후기 가속을 보여주는 탐색적 변화점이며 물리적 knee 정답은 아니다. 평활화·구간에 민감하고 전체 수명 정보를 사용하므로 예측 피처에서 제외한다.','small')

page('Q3  ΔQ(V)의 퍼짐이 수명과 일관되게 연관된다')
fig('04_delta_curves',maxheight=245)
table([['Batch','주분석 n','Pearson r','Spearman ρ','정책 군집 bootstrap 95% 구간'],*[[x['batch'],x['n_cells'],n(x['pearson'],3),f"{r('log_dq_var',x['batch'],'spearman'):.3f}",f"[{n(x['bootstrap_lo'],3)}, {n(x['bootstrap_hi'],3)}]"] for x in CI]],[95,90,125,125,CW-435])
txt('계산: ΔQ(V)=Q100(V)-Q10(V). 같은 1,000개 전압점에서 차이를 구하고 log10(표본분산)을 계산했다. 전압축을 정렬해 면적의 부호도 일관되게 처리했다.')
txt('관찰·시사점: 로그분산은 세 Batch 모두 음의 관계를 보이며 핵심 피처 후보이다. 구간은 정책 단위 부트스트랩 1,000회(난수 42)로 계산했으며, 적은 정책 수 때문에 탐색적으로 해석한다.')

page('Q3  수명 집단 비교와 통계 피처 선정')
fig('05_delta_groups',maxheight=240)
table([['피처','계산 / 해석','모델 입력 결정'],['log_dq_var','ΔQ 표본분산(ddof=1)의 log10','주피처'],['min / mean / max / std','전압축에서 ΔQ의 위치와 퍼짐','중복과 추가 가치를 내부 CV로 확인'],['skew / kurtosis','왜도와 초과첨도, bias=False','로그분산보다 약한 상관으로 후순위'],['signed area','전압 오름차순에서 ΔQ 적분 (Ah·V)','평균과 거의 중복, 우선 제외']],[150,390,CW-540])
txt('관찰: Batch 2에서 단수명(<500)과 장수명(>1000)을 비교할 수 있다. Batch 1·3에는 <500 셀이 없어 중간/장수명만 표시했다. 선은 중앙값, 음영은 IQR이다.','small')
txt('해석·시사점: 그룹 곡선의 차이는 후보 신호를 보여준다. 그래프의 분리 정도나 상관계수만으로 예측 정확도를 주장하지 않고 DAY 2 검증으로 확인한다.','small')

page('Q3  절대 단수명이 없는 Batch는 상대 비교로 보완한다')
fig('13_delta_relative_groups',maxheight=230)
relative=read('delta_relative_groups')
table([['Batch','수명 하위 25%: n / 범위','수명 상위 25%: n / 범위','log ΔQ 분산 중앙값: 하위 / 상위'],
 *[[b,*[x['n']+' / '+n(x['life_min'],0)+'~'+n(x['life_max'],0) for x in relative if x['batch']==b],
 ' / '.join(n(x['median_log_dq_var'],3) for x in relative if x['batch']==b)] for b in ['Batch 1','Batch 2','Batch 3']]],[85,195,195,CW-475])
txt('관찰: 각 Batch 내부의 수명 하위·상위 집단을 비교해도 하위 집단의 로그분산 중앙값이 더 높다. 선은 중앙값, 음영은 IQR이며 신뢰구간이 아니다. 경계 동률은 포함하므로 정확히 25%가 아닐 수 있다.')
txt('시사점: 절대 기준의 빈 집단을 가짜로 만들지 않고 Batch 내 상대 차이로 보완한다. 이 집단 구분은 전체 수명을 이용한 EDA 전용이며 예측 입력에는 넣지 않는다.')

for b,comment in [('1','관찰: 정책당 표본은 대체로 1~3개이다. 평균 순위 차이가 보여도 일부 장수명 정책은 종료 미확인 셀을 포함하므로 이 순위를 확정 수명 순위로 사용하지 않는다.'),('2','관찰: 4.8C(80%)-4.8C는 일반 조건 483.6(n=5), newstructure 871.7(n=3)이다. 동일한 C-rate 표기만으로 수명 차이를 설명할 수 없다.'),('3','관찰: 정책당 수명 편차가 있고, 두 단계 등가 C-rate는 약 4.794~4.813으로 좁다. 제한된 범위에서 속도-수명 관계가 약하다는 결과를 넓은 C-rate 범위에 일반화하지 않는다.')]:
 page(f'Q4  Batch {b} 충전 정책별 수명')
 fig('07_policy_batch'+b,width=600,maxheight=365)
 txt(comment)
 txt('막대는 정답이 있는 전체 셀의 평균, 오차막대는 ±1 표준편차이다(n=1은 추정 불가). (new)는 원문의 -newstructure를 뜻하며 다른 정책으로 유지했다. 단위: 사이클.','small')

page('Q4  충전 속도와 열화의 관계는 Batch에 의존한다')
fig('08_rate_and_fade',maxheight=235)
data=[['Batch','C_eq-수명 r','C_eq-초기기울기 r','실측 전류-초기기울기 r']]
for b in ['Batch 1','Batch 2','Batch 3']:
 rr=[x for x in R if x['batch']==b]
 data.append([b,f"{r('c_equiv',b):+.3f}",n(next(x['pearson'] for x in rr if x['x']=='c_equiv'),3),n(next(x['pearson'] for x in rr if x['x']=='current10_p95_A'),3)])
table(data,[95,180,230,CW-505])
txt('C_eq = 0.8 / [p/C1 + (0.8-p)/C2], p=전환 SOC/100. 0~80% SOC의 두 단계 등가 속도이며 CV·휴지 구간을 포함한 실제 충전시간과 다르다. 실측 전류는 cycle 10의 양의 I 값의 95백분위수이다.')
txt('해석·시사점: 빠를수록 항상 짧다는 전역 규칙은 지지되지 않는다. 정책·구조·배치가 함께 달라질 수 있으므로 정책 그룹 검증과 보조 피처 비교로 반영한다.')

page('Q4  명목 C-rate와 실제 전류 패턴을 구분한다')
fig('14_current_patterns',maxheight=255)
txt('관찰: cycle 10의 충전 전류를 Batch별 최단·중앙 순위·최장 셀에서 직접 확인했다. 충전 단계 전환과 후반의 전류 감소는 하나의 C-rate 수치가 생략하는 시간 정보를 보여준다. 그래프는 첫 양의 전류부터 첫 음의 방전 전류 직전 구간이다.')
table([['관측량','해석 가능한 정보','해석의 한계'],
 ['정책 C1 / 전환 SOC / C2','설정된 두 단계 급속충전 조건','후반 정전압 충전·휴지 시간을 포함하지 않는다.'],
 ['실측 양의 전류 P95','cycle 10 전류 크기의 상단을 요약','시간 가중 평균·에너지·전체 패턴을 대신하지 않는다.'],
 ['전류 P95-초기 QD 기울기 r','Batch별 -0.016 / -0.404 / -0.374','설정 조건과 구조 차이가 섞여 인과 추정은 아니다.']],[190,260,CW-450])
txt('시사점: P95는 패턴의 보조 진단 지표로 유지한다. 본 모델은 ΔQ 대표량부터 시작하고 충전 조건의 추가 가치는 정책 그룹 내부 검증으로 판단한다.')

page('Q5  Batch별 상관과 순위 상관을 함께 비교했다')
fig('09_correlations',maxheight=367)
txt('관찰: ΔQ 로그분산은 일관되지만 충전시간은 +0.571/-0.919/+0.599로 부호가 바뀐다. 온도와 초기 QD 기울기도 배치별 강도·방향이 다르다. Pooled 열은 Batch 차이가 섞인 값이다.')
txt('시사점: 가장 강한 pooled 상관 하나로 피처를 선정하지 않는다. 예측에 쓰는 후보는 Batch 1 내부 검증으로 비교하고, Batch 2·3 관찰은 일반화 위험을 설명하는 데 사용한다.','small')

page('Q5  중복 피처를 줄여 작은 표본에 맞게 설계했다')
fig('10_collinearity',width=CW,maxheight=295)
pairs=read('feature_pair_correlations')
pair_rows=[['Batch','log_dq_var vs dq_std','dq_mean vs dq_area','Tavg vs Tmax']]
for b in ['Batch 1','Batch 2','Batch 3']:
 vals=[]
 for a,z in [('log_dq_var','dq_std'),('dq_mean','dq_area'),('tavg_median','tmax_median')]:
  vals.append(n(next(x['pearson'] for x in pairs if x['batch']==b and {x['feature_x'],x['feature_y']}=={a,z}),3))
 pair_rows.append([b,*vals])
table(pair_rows,[95,230,230,CW-555],padding=4)
txt('관찰: Batch 1에서 Tavg-Tmax r=0.949, log_dq_var-dq_std r=0.991, ΔQ 평균-면적 r≈1이다. 특히 log10 Var=2log10 Std이므로 분산과 표준편차는 독립 정보가 아니다.')
txt('시사점: log_dq_var를 대표로 시작한다. 온도는 Tavg 하나, ΔQ 계열은 대표량 우선으로 구성하고 Ridge/ElasticNet의 규제를 사용한다. 상관이 약하다는 이유만으로 변수의 물리적 영향이 없다고 단정하지 않는다.')

page('Q5  핵심 피처의 분포 이동도 검증 설계에 반영한다')
fig('12_feature_distributions',maxheight=340)
txt('관찰: ΔQ 로그분산·충전시간·온도·저항 등은 Batch별 분포가 다르고, Batch 3의 등가 C-rate는 매우 좁은 범위에 모인다. 박스는 중앙값과 IQR, 점은 개별 셀이다. 수치 요약과 결측 수는 feature_distributions.csv에 저장했다.')
txt('시사점: 전체 데이터를 먼저 표준화하지 않는다. 학습 fold에서만 중앙값 대치·표준화를 적합하며, 외부 Batch의 범위 이탈과 오차를 함께 보고한다. 단일 배치에서 변동이 작다고 모든 환경에서 무의미한 변수라 결론내리지 않는다.')

page('Q5  최고 상관 순위와 정제 민감도를 함께 확인한다')
strong=[]
for b in ['Batch 1','Batch 2','Batch 3']:
 g=[x for x in C if x['batch']==b and x['scope']=='quality_screened']
 a=max(g,key=lambda x:abs(float(x['pearson'])));z=max(g,key=lambda x:abs(float(x['spearman'])))
 strong.append([b,a['feature']+' / '+n(a['pearson'],3),z['feature']+' / '+n(z['spearman'],3)])
table([['Batch','절대 Pearson 최강 피처 / r','절대 Spearman 최강 피처 / ρ'],*strong],[95,330,CW-425])
table([['Batch','log ΔQ 분산: 전체 유효 정답 r','품질 점검 후 r','해석'],
 *[[b,n(r('log_dq_var',b,scope='all_labeled'),3),n(r('log_dq_var',b),3),'제외 규칙 적용 전후의 방향·강도 비교'] for b in ['Batch 1','Batch 2','Batch 3']]],[95,205,155,CW-455])
txt('해석: 배치마다 최고 순위의 피처가 같을 필요는 없다. 주피처는 한 배치의 최대 상관만으로 고르지 않고 여러 배치에서의 방향, 중복성, 초기 관측 가능성과 단순성을 함께 고려한다.')
txt('시사점: log_dq_var를 대표 피처로 시작한다. 여기서 상관을 확인한 것은 모델 성능 검증이 아니다. 제외 규칙에 따른 선택 편향과 배치 효과를 남은 한계로 명시하고, 새 미관찰 배치로 확인한다.')

page('03  EDA 근거를 피처 집합과 타깃으로 연결했다')
txt('선택: 초기 100사이클 기반 회귀. 타깃은 제공 cycle_life이며, log10 타깃 변환을 비교하고 평가 때 원래 사이클 단위로 복원한다. 분류는 Batch 1의 500 미만 표본이 없어서 선택하지 않았다.')
table([['집합 / 변수','EDA 근거','설계 결정'],['F1: log_dq_var','세 Batch에서 음의 상관 방향 유지','한 피처 회귀를 기준으로 시작'],['F2: F1 + qd_slope + c_equiv','초기 변화·충전 조건의 보완 가능성, 배치별 관계 차이','Batch 1 내부 CV에서 추가 가치가 있는지 비교'],['F3: F2 + tavg_median + ir_median','단독 관계는 약하지만 조합 효과 가능','제한된 민감도 실험'],['dq_std / dq_mean / dq_area / Tmax 동시 투입','대표 ΔQ/온도 피처와 높은 공선성','우선 제외'],['충전시간 / 정책 원핫 / newstructure','부호 변화, 희소 범주, 학습 배치에 없는 조건','우선 보류; 그룹·오류 분석으로 추적'],['cell_id, batch, n_cycles, last_qd, knee','식별자 또는 미래·정답 정보','입력 금지']], [250,280,CW-530])
txt('정제 전후 상관은 correlations.csv에 scope=all_labeled / quality_screened로 함께 저장했다. 관측 종료와 채널 품질 조건을 바꾸면 결과가 달라질 수 있으므로 향후 민감도 결과도 보고한다.','small')

page('04  후보 모델은 데이터 특성과 연결해 비교한다')
table([['후보','왜 이 모델인가?','DAY 2 비교 설정'],['중앙값 DummyRegressor','학습이 단순 수명 평균/중앙값보다 유용한지 확인','원래 cycle_life 중앙값 기준'],['Ridge 우선','36개 학습 후보, 중복 피처와 작은 표본에 규제 필요','alpha: 0.1, 1, 10, 100'],['ElasticNet','제한된 변수 선택과 계수 축소를 함께 비교','alpha: 0.001, 0.01, 0.1 / l1_ratio: 0.2, 0.8'],['얕은 RandomForest','비선형 비교군. 과적합과 수명 범위 외삽 한계 점검','n_estimators=300, max_depth=2~3, min_samples_leaf=3~5']],[160,335,CW-495])
txt('공통 전처리: 결측 중앙값 대치 → 필요한 모델에 표준화 → 모델 적합을 Pipeline으로 구성한다. 대치·표준화는 각 교차검증 fold의 학습 부분에만 fit한다. 학습 피처는 초기 사이클 화이트리스트로 고정한다.')
txt('선택 기준: Train 내부 그룹 CV의 MAPE 평균과 변동성을 우선 확인하고, 비슷한 성능이면 적은 피처·단순 모델을 선택한다. 소표본이므로 큰 하이퍼파라미터 탐색이나 딥러닝을 우선하지 않는다.')
txt('외삽 위험: 주분석 Batch 1 수명은 534~1,074이지만 Batch 2는 392, Batch 3은 1,935까지 포함한다. 특히 트리는 학습 타깃 범위 밖으로 일반화하기 어렵다. 낮은 평균 오차만으로 장수명 추정 신뢰성을 주장하지 않는다.')

page('05  검증 절차와 평가 한계를 명확히 고정한다')
table([['단계','설계','의미'],['1. 내부 분리','Batch 1 36셀 → 정책 그룹 단위 train 29 / validation 7. 정책 16 / 4, 난수 42','동일 정책과 동일 셀의 분리 간 중복 없음'],['2. 모델 선택','train 29셀에서 4-fold GroupKFold. 전처리는 fold 내부 fit','F1~F3와 좁은 후보군 비교'],['3. 내부 확인','선택 후 validation 7셀에서 한 번 확인','소표본 변동성과 과적합 확인'],['4. 외부 Batch 평가','절차 동결 → Batch 1 36셀 재학습 → Batch 2 39셀 / Batch 3 40셀 평가','테스트 성능을 보고 튜닝 반복하지 않음']], [120,390,CW-510])
txt('주지표: MAPE(%) = mean(|실제-예측|/실제)×100. 보조: MAE·RMSE(사이클), R². train/validation/Batch 2/Batch 3 성능표, 실제-예측 산점도, 잔차, 수명 구간·정책별 큰 오차를 보고한다.')
txt('평가의 한계: DAY 1 과제에서 Batch 2·3의 정답 분포와 상관을 이미 관찰했다. 향후 평가는 “EDA에서 관찰한 외부 Batch 평가”이며 완전한 블라인드 테스트가 아니다. Batch 1 validation 역시 전체 EDA 후 설계됐다. 독립적인 최종 검증에는 새 미관찰 배치가 필요하다.')
txt('논문 9.1%는 다른 데이터 정제·분할 조건의 참고 수치이다. 이번 실험의 측정 성능이나 재현 보장으로 사용하지 않는다. [1]','small')

page('06  재현 방법, 활용 범위와 제출 점검')
table([['산출물','역할'],['notebooks/01_data_check.ipynb','파일 구조·품질·제외 이유·원본 SHA-256·라이브러리 버전'],['notebooks/02_day1_eda.ipynb','다섯 질문의 그래프·표·해석·모델 설계, 실행 출력 포함'],['src/day1_analysis.py','선택적 HDF5 로더, 피처 계산, 그림·표 생성 함수'],['results/day1/figures · tables','고해상도 PNG 17개, 셀 피처·품질·정책·상관·분리 계획 CSV'],['scripts/build_report.py','분석 표·그림에서 동일 PDF 재생성']], [305,CW-305])
txt('실행: 프로젝트 .venv 커널을 선택해 01 → 02 순서로 Restart & Run All 한다. PDF는 python scripts/build_report.py로 다시 만든다. requirements-day1.txt와 requirements-day1.lock.txt에서 실행 패키지와 검증 버전을 확인한다.')
txt('ESS 활용: 셀 수명 선별·교체 우선순위 연구에 활용할 수 있다. 실제 ESS는 온도, 부하, 달력 열화, 팩 불균형이 달라 외부 운전 데이터 검증이 필요하다. 입력 분포 변화와 정답 확보 후 오차 증가를 모니터링하고 새 배치 검증 후 재학습한다.')
txt('제출 전 확인: 과제의 다섯 질문 및 Batch 비교, 관찰-해석-시사점, 한 가지 모델링 방식, 피처 선정과 모델 후보 연결을 모두 포함했다. 개인 제출자 정보는 SKALA 3반 U095 이연주이다.')
txt('AI 도구를 활용해 코드·분석·문서 초안을 작성했다. 최종 제출 전 제출자가 결과와 한계를 읽고 설명할 수 있도록 노트북에 학습용 해설을 포함했다.','small')

page('참고자료와 요구사항 대응표')
table([['과제 요구사항','대응 산출물'],['수명 분포, 장/단수명 비율, 이상 셀','Q1 및 batch_summary.csv, cell_audit.csv'],['열화 곡선, 가속, knee','Q2 및 대표 셀 표, cell_features.csv의 knee_*'],['ΔQ(V) 차이, 장/단수명 비교, 통계 피처','Q3 및 delta_correlation_uncertainty.csv'],['충전 정책별 수명, 고속충전, 전류-열화 관계','Q4 및 policy_summary.csv, rate_fade_correlations.csv'],['피처 상관, 강한 관계, 다중공선성','Q5 및 correlations.csv, 10_collinearity.png'],['피처/타깃 선정, 모델 후보와 EDA 연결','모델 설계 03~05 및 planned_split.csv']], [340,CW-340])
txt('[1] Severson, K. A., Attia, P. M., Jin, N., et al. (2019). Data-driven prediction of battery cycle life before capacity degradation. Nature Energy, 4, 383-391. DOI: 10.1038/s41560-019-0356-8.','small')
txt('[2] 저자 공개 데이터 처리 코드: github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation (Load Data.ipynb, LoadData.m, README). 2026-10-01 확인. 모델 코드를 복사하지 않고 분석 함수를 직접 구현했다.','small')
txt('[3] 교수님 제공 30-ESSHealth-scratch.ipynb, DS Mini Project 강의 webarchive, 사용자 첨부 DAY 1 과제 화면 3장. 과제의 5개 질문과 설계 요구사항을 기준으로 구성했다.','small')
txt('[4] 제공 원본: 2017-05-12 / 2018-02-20 / 2018-04-12 batchdata_updated_struct_errorcorrect.mat. 데이터 지문은 results/day1/tables/data_fingerprints.csv에 저장했다. 개인 진행 HTML은 참고 흐름으로 사용했다.','small')

page('최종 점검  |  평가 항목과 제출 근거')
table([['평가 항목','충족 근거','검증 범위'],
 ['EDA · 50점','Q1~Q5, Batch 1·2·3 비교, 핵심 피처 분포, Pearson/Spearman, 공선성, 피처 추출','원본 139셀 / 주분석 115셀을 구분하고 실행 결과로 확인'],
 ['EDA → 전략 · 30점','각 질문의 관찰·해석·시사점, 상대 ΔQ 비교, 이상 셀 상대 위치, 정제 민감도','ΔQ 신호 → 대표 피처 / 중복 → 규제 / 분포 이동 → 외부 배치 평가'],
 ['모델 전략 · 20점','회귀 타깃, F1~F3, Dummy/Ridge/ElasticNet/얕은 RF의 선택 이유','학습 전처리·그룹 CV·평가 지표·외삽 한계를 명시']],[150,355,CW-505])
txt('Scratch 보존: 새 노트북은 src의 독립 로더로 MAT 원본을 읽는다. 교수님 Scratch의 변수·그래프·CSV·실행 순서에 의존하지 않는다. 따라서 Scratch의 해석·표시상 주의점을 제출 분석에 전파하지 않고 원래 파일을 보존했다.')
txt('제출 범위: DAY 1은 분석과 모델 설계이다. 모델 적합이나 정확도 측정 결과를 만들어 넣지 않았다. 모든 결과는 해당 제공 파일과 명시한 정제 규칙에 한정한다. 요구사항 대응 여부는 점검했지만 실제 점수는 평가자가 판단한다.')
txt('재현 순서: 01_data_check → 02_day1_eda → build_report → verify_day1. 새 노트북의 코드와 출력, 그림·표, 실행 기록을 함께 보관한다. 자동 재생성 스크립트는 새 노트북을 덮어쓰므로 수동 수정 전 백업한다.','small')

def footer(c,doc):
 c.saveState()
 c.setFillColor(NAVY);c.rect(0,H-9,W,9,fill=1,stroke=0)
 c.setFillColor(GREEN);c.rect(0,H-9,105,9,fill=1,stroke=0)
 c.setFont('KR',7.5);c.setFillColor(GRAY)
 c.drawString(44,H-24,'ESS RESEARCH NOTE  /  2026.10.01')
 c.drawRightString(W-44,H-24,'DAY 1  ·  DATA → INSIGHT → STRATEGY')
 c.setStrokeColor(colors.HexColor('#D5E1EC'));c.line(44,31,W-44,31)
 c.setFont('KR',8);c.setFillColor(GRAY);c.drawString(44,18,'SKALA 3반 · U095 이연주  |  ESS Battery · DAY 1')
 c.setFillColor(GREEN);c.roundRect(W-78,12,34,18,4,fill=1,stroke=0);c.setFillColor(colors.white);c.drawCentredString(W-61,18,f'{doc.page:02d}');c.restoreState()
doc=SimpleDocTemplate(str(PDF),pagesize=(W,H),leftMargin=44,rightMargin=44,topMargin=42,bottomMargin=43,title='ESS 배터리 수명 예측 DAY 1 모델 전략 - SKALA 3반 U095 이연주',author='이연주')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(PDF)
