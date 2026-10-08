"""Bounded CPU Modal runner with durable per-split checkpoints."""
import os, subprocess, modal
app=modal.App('ddr-mksvm-v8-class-sensitive')
volume=modal.Volume.from_name('ddr-mksvm-v8-class-sensitive-results',create_if_missing=True)
image=(modal.Image.debian_slim(python_version='3.12').pip_install(
    'numpy==1.26.4','scipy==1.17.1','pandas==3.0.5','scikit-learn==1.9.0','joblib==1.6.0','cvxpy==1.6.7')
    .add_local_dir('ddr_mksvm',remote_path='/opt/project/ddr_mksvm',copy=True)
    .add_local_dir('dataset',remote_path='/opt/project/dataset',copy=True)
    .add_local_file('run_v8_class_sensitive.py',remote_path='/opt/project/run_v8_class_sensitive.py',copy=True)
    .add_local_file('modal_v8_class_sensitive.py',remote_path='/opt/project/modal_v8_class_sensitive.py',copy=True))

@app.function(image=image,cpu=4,memory=4096,timeout=3600,volumes={'/artifacts':volume})
def run(datasets,variants,n_seeds,seed_start,run_name,local_git,reuse_root):
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        V8_MODAL_RUN='1',V8_LOCAL_GIT=local_git,MODAL_APP_ID=app.app_id)
    try:
        cmd=['python','run_v8_class_sensitive.py','--datasets',datasets,'--variants',variants,
            '--n-seeds',str(n_seeds),'--seed-start',str(seed_start),'--run-name',run_name,'--output-root','/artifacts']
        if reuse_root:cmd+=['--reuse-root','/artifacts/'+reuse_root]
        subprocess.run(cmd,
            cwd='/opt/project',env=env,check=True)
    finally:volume.commit()
    return '/artifacts/'+run_name

@app.local_entrypoint()
def main(datasets:str='blood_transfusion,mammographicmass_binary,heart_disease,parkinson',
         variants:str='v7,selection_balanced,selection_f1,v8_a,v8_b,inverse_f1,v8_c,v8_d',
         n_seeds:int=1,seed_start:int=9000,run_name:str='smoke/v8-smoke-20260909',reuse_root:str=''):
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','status','--porcelain'],text=True)
    print(run.remote(datasets,variants,n_seeds,seed_start,run_name,commit+' dirty='+str(bool(dirty)),reuse_root))
