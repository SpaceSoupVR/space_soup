import subprocess, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import strip as strip2
repo=sys.argv[1]; base_rev, ours_rev, theirs_rev = sys.argv[2:5]
def git(*a, inp=None):
    return subprocess.run(['git','-C',repo,*a],capture_output=True,input=inp)
def files(rev):
    out=git('ls-tree','-r','--name-only',rev).stdout.decode().splitlines()
    return set(out)
def blob(rev,path):
    r=git('show',f'{rev}:{path}'); return r.stdout if r.returncode==0 else None
def code(path): return path.endswith(('.rs','.wgsl'))
def strip(b):
    if b is None: return None
    return strip2.strip(b.decode('utf-8')).encode()
B,O,T=files(base_rev),files(ours_rev),files(theirs_rev)
conf=[];out={}
for p in sorted(B|O|T):
    b,o,t=blob(base_rev,p) if p in B else None, blob(ours_rev,p) if p in O else None, blob(theirs_rev,p) if p in T else None
    if code(p): b,o=strip(b),strip(o)
    if o==t: out[p]=t; continue
    if o==b: out[p]=t; continue          # only theirs changed (incl. delete)
    if t==b: out[p]=o; continue          # only ours changed
    if o is None or t is None or b is None:
        if b is None and o is not None and t is None: out[p]=o; continue
        if b is None and t is not None and o is None: out[p]=t; continue
        conf.append((p,'add/delete')); out[p]=o if o is not None else t; continue
    with tempfile.TemporaryDirectory() as d:
        for n,v in (('o',o),('b',b),('t',t)): open(f'{d}/{n}','wb').write(v)
        r=subprocess.run(['git','merge-file','-L','ours','-L','base','-L','theirs',f'{d}/o',f'{d}/b',f'{d}/t'])
        out[p]=open(f'{d}/o','rb').read()
        if r.returncode!=0: conf.append((p,f'{r.returncode} hunks'))
# write work tree
for p,v in out.items():
    fp=os.path.join(repo,p)
    if v is None:
        if os.path.exists(fp): os.remove(fp)
        continue
    os.makedirs(os.path.dirname(fp) or '.',exist_ok=True); open(fp,'wb').write(v)
for p in (B|O)-set(k for k,v in out.items() if v is not None):
    fp=os.path.join(repo,p)
    if os.path.exists(fp): os.remove(fp)
print('CONFLICTS',len(conf))
for c in conf: print(' ',*c)
