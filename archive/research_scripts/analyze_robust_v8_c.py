import argparse
from pathlib import Path
from archive.research_scripts.run_robust_v8_c import aggregate
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("run_dir");p.add_argument("--output-dir");a=p.parse_args();root=Path(a.run_dir);destination=Path(a.output_dir) if a.output_dir else root/"analysis";_,_,s=aggregate(root,destination);print(s.to_string(index=False) if not s.empty else "No checkpoints")
