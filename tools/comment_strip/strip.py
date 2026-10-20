import sys, re
import re as _re
_TRAIL=_re.compile(r'^(.*?\S)\s+//(?!/).*$')
def _trail(l):
    body=l.rstrip('\n')
    m=_TRAIL.match(body)
    if not m: return l
    code=m.group(1)
    if code.count('"')%2: return l
    return code+'\n'
KEEP=('// <fx:','// </fx:')
def strip(text):
    out=[]
    for l in text.splitlines(keepends=True):
        if l.lstrip().startswith('//'):
            if l.lstrip().startswith(KEEP):
                out.append(l); continue
            l='\n'
        elif '//' in l:
            l=_trail(l)
        if l.strip()=='' and (not out or out[-1].strip()=='' or out[-1].rstrip().endswith('r#"')):
            continue
        out.append(l)
    while out and out[-1].strip()=='': out.pop()
    return ''.join(out)

if __name__=='__main__':
    import subprocess
    repo,a,b=sys.argv[1:4]
    files=subprocess.run(['git','-C',repo,'diff','--name-only',a,b],capture_output=True,text=True).stdout.split()
    ok=bad=0
    for f in files:
        if not f.endswith(('.rs','.wgsl')): continue
        try:
            src=subprocess.run(['git','-C',repo,'show',f'{a}:{f}'],capture_output=True,text=True,check=True).stdout
            dst=subprocess.run(['git','-C',repo,'show',f'{b}:{f}'],capture_output=True,text=True,check=True).stdout
        except subprocess.CalledProcessError: continue
        s=strip(src)
        if s==dst: ok+=1
        else:
            bad+=1
            import difflib
            d=[x for x in difflib.unified_diff(s.splitlines(),dst.splitlines(),lineterm='',n=0) if x[:1] in '+-' and not x.startswith(('+++','---'))]
            print(f, len(d), d[:4])
    print('ok',ok,'bad',bad)
