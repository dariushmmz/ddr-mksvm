"""Independent V8.1 provenance and write-once artifact helpers; no fitting."""
from pathlib import Path
import hashlib
import json
import argparse

ROOT=Path('results/v8_1/analysis')
DOCS=['docs/V8_1_ASTRA_INVESTIGATION.md','docs/V8_1_MATHEMATICAL_DESIGN.md','docs/V8_1_EXPERIMENT_PLAN.md']


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
    if path.exists():
        assert path.read_bytes()==payload,f'Refusing overwrite: {path}'
    else:
        with path.open('xb') as stream:stream.write(payload)


def verify_baselines():
    reference=json.loads((ROOT/'baseline_integrity.json').read_text())
    for path,expected in reference['hashes'].items():assert digest(path)==expected,path
    return len(reference['hashes'])


def freeze_design():
    verify_baselines()
    old=json.loads(Path('results/v8/analysis/deterministic_v8_c_final_freeze.json').read_text())
    value=dict(date='2026-09-15',status='design_frozen_before_candidate_implementation',
        candidate='v8_1_kernel_choice',candidate_families=1,robust_work='blocked',
        document_hashes={p:digest(p) for p in DOCS},baseline_code_hashes=old['code_hashes'],
        baseline_freeze_hash=digest('results/v8/analysis/deterministic_v8_c_final_freeze.json'),
        dataset_inventory=old['configuration']['dataset_inventory'],versions=old['configuration']['versions'],
        development_seeds=list(range(11000,11024)),confirmation_seeds=list(range(12000,12096)),
        confirmation_status='untouched; requires successful gate24 and separate candidate_freeze.json')
    write_new(ROOT/'design_freeze.json',value)
    print('Design frozen:',digest(ROOT/'design_freeze.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze-design',action='store_true');a=p.parse_args()
    if a.freeze_design:freeze_design()
    else:print('Frozen baseline files verified:',verify_baselines())
