"""Apply prospectively fixed deterministic confirmation criteria; no fitting."""
import argparse, json
from pathlib import Path
import pandas as pd
from ddr_mksvm.experiments.class_sensitive_runner import json_write

GUARDS={'parkinson':.01,'wine':.005,'iris':.005,'breast_cancer_diagnostic':.005}

def analyze(root):
    root=Path(root)
    cfg=json.loads((root/'config.json').read_text())
    freeze=json.loads(Path('results/v8/analysis/deterministic_candidate_freeze.json').read_text())
    manifest=json.loads((root/'manifest.json').read_text())
    assert manifest['status']=='complete'
    assert cfg['seeds']==list(range(10000,10096))
    assert cfg['variants']==['v7','v8_c']
    assert cfg['code_hashes']==freeze['code_hashes']
    frame=pd.read_csv(root/'per_run.csv');paired=pd.read_csv(root/'paired_comparisons.csv')
    assert set(frame.dataset)==set(cfg['datasets'])
    assert set(frame.model)==set(cfg['variants'])
    assert len(frame)==len(cfg['datasets'])*len(cfg['variants'])*96
    assert (frame.groupby(['dataset','model']).size()==96).all()
    def row(ds,metric):
        return paired[(paired.dataset==ds)&(paired.metric==metric)&(paired.model=='v8_c')].iloc[0]
    primary=row('blood_transfusion','balanced_accuracy')
    secondary=row('heart_disease','macro_f1')
    guard_rows={ds:row(ds,'test_error') for ds in GUARDS}
    failed=[ds for ds,rec in guard_rows.items() if rec.mean_delta>GUARDS[ds]]
    primary_pass=bool(primary.mean_delta>=.01 and primary.ci_low>0)
    secondary_pass=bool(secondary.mean_delta>=.01 and secondary.ci_low>0)
    result=dict(primary_endpoint='Blood balanced accuracy',primary_pass=primary_pass,
        secondary_endpoint='Heart macro F1',secondary_pass=secondary_pass,
        guard_mean_error_failures=failed,
        guard_noninferiority_not_established=[ds for ds,rec in guard_rows.items() if rec.ci_high>GUARDS[ds]],
        passes_prespecified_point_screen=bool(primary_pass and not failed),
        interpretation='Point-screen qualification is not proof of noninferiority or a universal accuracy improvement.',
        original_contribution='Standard weighted SVM and balanced CV; novelty is not established by this experiment.',
        fingerprint=manifest['fingerprint'],app_id=manifest['app_id'])
    out=root/'qualification.json'
    if out.exists():assert json.loads(out.read_text())==result
    else:json_write(out,result)
    print(json.dumps(result,indent=2))
    print(paired[['dataset','metric','mean_delta','ci_low','ci_high','wins','ties','losses','wilcoxon_p','standardized_paired_effect','runtime_ratio']].to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();analyze(a.root)
