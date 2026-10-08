"""Bounded, deterministic-only V8.1 Modal workflow."""
import os
import subprocess
import modal

app=modal.App('ddr-mksvm-v8-1-deterministic')
volume=modal.Volume.from_name('ddr-mksvm-v8-1-results',create_if_missing=True)
image=(modal.Image.debian_slim(python_version='3.12').pip_install(
    'numpy==1.26.4','scipy==1.17.1','pandas==3.0.5','scikit-learn==1.9.0','joblib==1.6.0','cvxpy==1.6.7')
    .add_local_dir('ddr_mksvm',remote_path='/opt/project/ddr_mksvm',copy=True)
    .add_local_dir('dataset',remote_path='/opt/project/dataset',copy=True))
for path in ['run_v8_class_sensitive.py','modal_v8_class_sensitive.py','run_v8_1.py','modal_v8_1.py','v8_1_provenance.py',
    'docs/V8_1_ASTRA_INVESTIGATION.md','docs/V8_1_MATHEMATICAL_DESIGN.md','docs/V8_1_EXPERIMENT_PLAN.md',
    'results/v8_1/analysis/design_freeze.json']:
    image=image.add_local_file(path,remote_path='/opt/project/'+path,copy=True)


@app.function(image=image,cpu=4,memory=4096,timeout=1200,volumes={'/artifacts':volume})
def run(stage,run_name,reuse_root,local_git,gate_authorization):
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        V8_1_MODAL_RUN='1',MODAL_APP_ID=app.app_id,V8_1_LOCAL_GIT=local_git)
    cmd=['python','run_v8_1.py','--stage',stage,'--run-name',run_name,'--output-root','/artifacts']
    if reuse_root:cmd+=['--reuse-root','/artifacts/'+reuse_root]
    if gate_authorization:cmd+=['--gate-authorization','/artifacts/'+gate_authorization]
    try:subprocess.run(cmd,cwd='/opt/project',env=env,check=True)
    finally:volume.commit()
    return '/artifacts/'+run_name


@app.local_entrypoint()
def main(stage:str='smoke',run_name:str='smoke/v8-1-smoke-20260915',reuse_root:str='',gate_authorization:str=''):
    from v8_1_provenance import verify_baselines
    print('Frozen baseline files checked:',verify_baselines())
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True))
    print(run.remote(stage,run_name,reuse_root,head+' dirty='+str(dirty),gate_authorization))
