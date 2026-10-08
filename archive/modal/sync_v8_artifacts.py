"""Non-destructive Modal artifact mirror with explicit Windows path handling."""
import argparse, hashlib, io, os, tarfile
from pathlib import Path, PurePosixPath
import modal
from concurrent.futures import ThreadPoolExecutor
from modal.volume import FileEntryType

def sync(remote, destination, bundle=False):
    volume=modal.Volume.from_name('ddr-mksvm-v8-class-sensitive-results')
    root=Path(destination).resolve();root.mkdir(parents=True,exist_ok=True)
    if bundle:
        data=b''.join(volume.read_file(remote.rstrip('/')+'/artifacts.tar.gz'))
        with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
            members=archive.getmembers()
            for member in members:
                target=(root/member.name).resolve()
                if not target.is_relative_to(root) or not member.isfile():raise ValueError('unsafe archive member')
                payload=archive.extractfile(member).read();target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists():
                    if target.read_bytes()!=payload:raise FileExistsError(target)
                else:
                    tmp=target.with_suffix(target.suffix+'.download');tmp.write_bytes(payload);os.replace(tmp,target)
        print(f'Mirrored {len(members)} files from {len(data)} compressed bytes to {root}');return
    entries=[e for e in volume.iterdir(remote,recursive=True) if e.type==FileEntryType.FILE]
    def copy_entry(entry):
        relative=PurePosixPath(entry.path.lstrip('/')).relative_to(remote.strip('/'))
        target=(root/str(relative)).resolve()
        if not target.is_relative_to(root):raise ValueError('invalid remote path')
        target.parent.mkdir(parents=True,exist_ok=True)
        data=b''.join(volume.read_file(entry.path))
        assert len(data)==entry.size
        if target.exists():
            if target.read_bytes()!=data:raise FileExistsError(f'incompatible local artifact {target}')
        else:
            tmp=target.with_suffix(target.suffix+'.download');tmp.write_bytes(data);os.replace(tmp,target)
        return entry.path
    with ThreadPoolExecutor(max_workers=4) as pool:
        copied=list(pool.map(copy_entry,entries))
    print(f'Mirrored {len(copied)} files to {root}')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('remote');p.add_argument('destination');p.add_argument('--bundle',action='store_true');a=p.parse_args();sync(a.remote,a.destination,a.bundle)
