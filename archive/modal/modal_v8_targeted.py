"""Modal entrypoint for the isolated V8 targeted-development runner."""
from pathlib import Path
import os, subprocess, modal
app=modal.App('ddr-mksvm-iris-v8-targeted')
volume=modal.Volume.from_name('ddr-mksvm-iris-results',create_if_missing=True)
image=(modal.Image.debian_slim(python_version='3.12').pip_install('numpy>=1.24,<2.0','scipy>=1.10','pandas>=1.5','scikit-learn>=1.2')
       .add_local_dir('ddr_mksvm',remote_path='/opt/ddr_project/ddr_mksvm',copy=True)
       .add_local_dir('dataset',remote_path='/opt/ddr_project/dataset',copy=True)
       .add_local_file('run_v8_targeted.py',remote_path='/opt/ddr_project/run_v8_targeted.py',copy=True)
       .add_local_file('analyze_v7_forensics.py',remote_path='/opt/ddr_project/analyze_v7_forensics.py',copy=True))
@app.function(image=image,cpu=4,memory=8192,timeout=3600,volumes={'/artifacts':volume})
def run(n_seeds:int,variants:str,run_name:str,seed_start:int=4000):
    cmd=['python','run_v8_targeted.py','--n-seeds',str(n_seeds),'--seed-start',str(seed_start),'--variants',variants,'--output',f'/artifacts/{run_name}']
    env=os.environ.copy();env.update({'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    subprocess.run(cmd,cwd='/opt/ddr_project',env=env,check=True);volume.commit();return f'/artifacts/{run_name}'
@app.function(image=image,cpu=4,memory=8192,timeout=3600,volumes={'/artifacts':volume})
def forensic(run_name:str):
    cmd=['python','analyze_v7_forensics.py','--v7-dir','/artifacts/iris-v7-rkhs-l2-final96-20260908','--output',f'/artifacts/{run_name}','--dataset','/opt/ddr_project/dataset/iris_multiclass.csv']
    subprocess.run(cmd,cwd='/opt/ddr_project',check=True);volume.commit();return f'/artifacts/{run_name}'
@app.local_entrypoint()
def main(n_seeds:int=1,variants:str='raw,pair_alpha,pair_C,pair_both',run_name:str='iris-v8-targeted',seed_start:int=4000):
    print(run.remote(n_seeds,variants,run_name,seed_start))
