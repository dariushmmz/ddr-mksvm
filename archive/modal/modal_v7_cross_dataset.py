"""CPU-only Modal wrapper for frozen V7 cross-dataset gates."""
import os,subprocess,modal
app=modal.App('ddr-mksvm-v7-cross-dataset');volume=modal.Volume.from_name('ddr-mksvm-v7-cross-dataset-results',create_if_missing=True)
image=(modal.Image.debian_slim(python_version='3.12').pip_install('numpy>=1.24,<2','scipy>=1.10','pandas>=1.5','scikit-learn>=1.2','joblib>=1.2','cvxpy>=1.6.5,<1.7')
 .add_local_dir('ddr_mksvm',remote_path='/opt/project/ddr_mksvm',copy=True).add_local_dir('dataset',remote_path='/opt/project/dataset',copy=True)
 .add_local_file('run_v7_cross_dataset.py',remote_path='/opt/project/run_v7_cross_dataset.py',copy=True))
@app.function(image=image,cpu=4,memory=8192,timeout=60*60*6,retries=modal.Retries(max_retries=1,initial_delay=5),volumes={'/artifacts':volume})
def run(datasets:str,n_seeds:int,seed_start:int,run_name:str,n_jobs:int=4):
 env=os.environ.copy();env.update({'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','V7_MODAL_RESOURCES':'4 CPU; 8192 MiB; no GPU'})
 cmd=['python','run_v7_cross_dataset.py','--datasets',datasets,'--n-seeds',str(n_seeds),'--seed-start',str(seed_start),'--n-jobs',str(n_jobs),'--run-name',run_name,'--output-root','/artifacts']
 subprocess.run(cmd,cwd='/opt/project',env=env,check=True);volume.commit();return f'/artifacts/{run_name}'
@app.local_entrypoint()
def main(datasets:str='parkinson',n_seeds:int=1,seed_start:int=7000,run_name:str='v7-cross-smoke',n_jobs:int=4):print(run.remote(datasets,n_seeds,seed_start,run_name,n_jobs))
