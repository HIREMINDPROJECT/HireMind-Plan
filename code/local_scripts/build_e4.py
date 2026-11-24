"""Build the E4 comparison table and the ablation table from verified run outputs (validation only)."""
import json,glob,csv,statistics as st,urllib.request
K=['accuracy','macro_f1','good_fit_recall','good_fit_precision','potential_fit_recall']
def jl(pat):
    o={}
    for f in glob.glob(pat,recursive=True):
        try: j=json.load(open(f))
        except Exception: continue
        if isinstance(j,dict) and 'seed' in j and 'accuracy' in j: o[j['seed']]=j
    return o
def agg(d): return {k:(st.mean(d[s][k] for s in d), st.stdev([d[s][k] for s in d]) if len(d)>1 else None) for k in K}
def tab(path,name):
    for r in csv.DictReader(open(path)):
        if r['model']==name: return {'accuracy':float(r['accuracy_pct']),'macro_f1':float(r['macro_f1_pct']),'good_fit_recall':float(r['good_fit_recall_pct']),'good_fit_precision':float(r['good_fit_precision_pct']),'potential_fit_recall':float(r['potential_fit_recall_pct'])}
def tabsd(path,name):
    for r in csv.DictReader(open(path)):
        if r['model']==name: return {'accuracy':float(r['accuracy_pct']),'good_fit_recall':float(r['good_fit_recall_pct'])}
rows=[]
def add(model,labels,inp,seeds,vals,sd=None,note=''):
    rows.append(dict(model=model,train_labels=labels,input=inp,seeds=seeds,**{k:round(v,2) for k,v in vals.items()},sd_accuracy=(round(sd['accuracy'],2) if sd else ''),sd_gf_recall=(round(sd['good_fit_recall'],2) if sd else ''),note=note))
S1D0='/home/claude/cres/C_stage1_m1_m2_validation.csv'; S1D1='/home/claude/e4res/E4_D1_stage1_m1_m2_validation.csv'
S2D0='/home/claude/c2res2/C_stage2_m3_m6_validation.csv'; S2D1='/home/claude/e4res/E4_D1_stage2_m3_validation.csv'
for lab,p in (('D0',S1D0),('D1',S1D1)):
    add('M1 LogReg TF-IDF',lab,'full text (TF-IDF)','deterministic',tab(p,'M1_LogReg_C1'))
    add('M2 LinearSVC TF-IDF',lab,'full text (TF-IDF)','deterministic',tab(p,'M2_LinearSVC_C1'))
for lab,p in (('D0',S2D0),('D1',S2D1)):
    add('M3 XGBoost (B1 features)',lab,'structured B1 v3.1 features','8 (42-49)',tab(p,'M3_XGBoost_mean8'),tabsd(p,'M3_XGBoost_sd8'))
d0=jl('/home/claude/e0res/e0_s*.json'); d1=jl('/home/claude/e1res/e1d1_results/**/results/knowledge_run_*.json')
e2d0=jl('/home/claude/e2res/E2_run_s*.json'); e2d1=jl('/home/claude/e2d1res/e2_results/**/*.json')
for name,lab,inp,d in (('M4 BERT-Tiny + Knowledge v1 (E0)','D0','512-token pair + Knowledge line',d0),('M4 BERT-Tiny + Knowledge v1 (E1)','D1','512-token pair + Knowledge line',d1),
                       ('M4 + evidence input (E2)','D0','B4 evidence selection + Knowledge line',e2d0),('M4 + evidence input (E2)','D1','B4 evidence selection + Knowledge line',e2d1)):
    a=agg(d); add(name,lab,inp,'8 (42-49)',{k:a[k][0] for k in K},{k:a[k][1] for k in K})
add('M5 DistilBERT','D0','384-token pair','1 (42)',tab('/home/claude/deliv/C_M5/C_stage2_m5_validation.csv','M5_DistilBERT_s42'),note='single seed; 384 tokens (CPU budget)')
add('M6 MiniLM + LogReg','D0','sentence embeddings','deterministic',tab(S2D0,'M6_MiniLM_LogReg_C0.1'),note='not part of the team decision (DistilBERT only); reported for completeness')
rid=45257498571
for v in ('full','minus_evidence','minus_knowledge','minus_struct'):
    try:
        j=json.load(urllib.request.urlopen(f'https://raw.githubusercontent.com/HIREMINDPROJECT/HireMind-Plan/m8-results-{rid}-{v}/m8_results/run_{rid}/{v}/M8_{v}_s42.json'))
        add(f'M8 DistilBERT hybrid ({v})','D0','evidence/Knowledge/structured as per variant (512 tokens)','1 (42)',{k:j[k] for k in K},note=f"best epoch {j['best_epoch']}"+('; stopped after 2 epochs by the time guard' if j.get('stopped_early_by_wall_guard') else ''))
    except Exception as e: print('M8',v,'missing',e)
ref=next(r for r in rows if r['model'].startswith('M4') and r['train_labels']=='D0' and '(E0)' in r['model'])
for r in rows:
    elig = r['accuracy']>ref['accuracy'] and r['good_fit_recall']>ref['good_fit_recall'] and all(ref[k]-r[k]<=2.0 for k in ('good_fit_precision','macro_f1','potential_fit_recall'))
    r['eligible_vs_M4_D0']= 'reference' if r is ref else ('YES' if elig else 'no')
with open('E4_model_comparison_validation.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows: print(f"{r['model'][:40]:40s} {r['train_labels']} acc {r['accuracy']:6.2f} mF1 {r['macro_f1']:6.2f} GFR {r['good_fit_recall']:6.2f} GFP {r['good_fit_precision']:6.2f} PR {r['potential_fit_recall']:6.2f} {r['eligible_vs_M4_D0']}")
