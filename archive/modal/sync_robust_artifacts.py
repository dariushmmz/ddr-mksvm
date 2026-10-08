"""Non-destructive partial/final mirror from the shared robust Modal Volume."""
from __future__ import annotations
import argparse, os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
import modal
from modal.volume import FileEntryType

def sync(remote:str,destination:str):
    volume=modal.Volume.from_name("ddr-mksvm-robust-results")
    root=Path(destination).resolve();root.mkdir(parents=True,exist_ok=True)
    entries=[e for e in volume.iterdir(remote,recursive=True) if e.type==FileEntryType.FILE]
    def one(entry):
        relative=PurePosixPath(entry.path.lstrip("/")).relative_to(remote.strip("/"))
        target=(root/str(relative)).resolve()
        if not target.is_relative_to(root):raise ValueError("unsafe remote path")
        data=b"".join(volume.read_file(entry.path));assert len(data)==entry.size
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if target.read_bytes()!=data:raise FileExistsError(f"immutable local artifact differs: {target}")
        else:
            tmp=target.with_suffix(target.suffix+".download");tmp.write_bytes(data);os.replace(tmp,target)
        return str(relative)
    with ThreadPoolExecutor(max_workers=4) as pool:copied=list(pool.map(one,entries))
    print(f"Mirrored {len(copied)} files to {root}")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("remote");p.add_argument("destination");a=p.parse_args();sync(a.remote,a.destination)
