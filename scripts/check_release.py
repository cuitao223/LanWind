"""Audit the public source tree and staged files before publishing."""
import ast,hashlib,json,re,subprocess,sys
from pathlib import Path
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
DATA_SUFFIXES={'.nc','.nc4','.grib','.grib2','.grb','.grb2','.npy','.npz','.h5','.hdf5','.parquet','.csv','.zip','.7z','.tar','.tgz','.gz','.jsonl','.pkl','.pickle'}
PRIVATE_NAMES={'key'+'.txt','.env','id_'+'rsa','id_'+'ed25519','ops'+'.py'}
def main():
    files=[p for p in ROOT.rglob('*') if p.is_file() and not any(x in {'.git','.venv','__pycache__','runs','build','dist'} for x in p.relative_to(ROOT).parts)]
    findings=[]
    forbidden_words=['competi'+'tion','con'+'test','t3'+'-server','shared'+'-nvme',chr(36187)+chr(39064),chr(27604)+chr(36187),chr(21442)+chr(36187),chr(33719)+chr(22870)]
    for p in files:
        rel=p.relative_to(ROOT).as_posix()
        if p.suffix.lower() in DATA_SUFFIXES or '.7z.' in p.name.lower() or p.name.lower() in PRIVATE_NAMES:findings.append('Forbidden data/private file: '+rel)
        if p.suffix=='.pt' and rel!='weights/lanwind.pt':findings.append('Unapproved checkpoint: '+rel)
        if p.suffix in {'.py','.md','.yaml','.yml','.toml','.txt','.svg','.json'} or p.name in {'.gitignore','LICENSE'}:
            content=p.read_text(encoding='utf8')
            for word in forbidden_words:
                if word in content.lower():findings.append('Private context in '+rel)
            secret_pattern=r'gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}|'+'-----'+'BEGIN .*PRIVATE KEY'+'-----'
            if re.search(secret_pattern,content):findings.append('Potential secret in '+rel)
        if p.suffix=='.py':ast.parse(p.read_text(encoding='utf8'),filename=rel)
        if p.suffix=='.svg':ET.parse(p)
    for name,other in [('README.md','README.en.md'),('README.en.md','README.md')]:
        text=(ROOT/name).read_text(encoding='utf8')
        if other not in text[:200]:findings.append('Missing top language switch: '+name)
        if text.count('```')%2:findings.append('Unbalanced code fences: '+name)
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)|(?:href|src)="([^"]+)"',text):
            target=next(t for t in target if t)
            if target.startswith(('http:','https:','#')):continue
            if not (ROOT/target.split('#')[0]).is_file():findings.append('Broken documentation link: '+target)
    tracked=subprocess.run(['git','ls-files','-z'],cwd=ROOT,capture_output=True).stdout.decode('utf8').split('\0')
    for name in filter(None,tracked):
        if Path(name).suffix.lower() in DATA_SUFFIXES or '.7z.' in name.lower() or any(x in {'data','datasets','samples','replays'} for x in Path(name).parts):findings.append('Data in Git index: '+name)
    import torch
    weights=torch.load(ROOT/'weights/lanwind.pt',map_location='cpu',weights_only=True)
    if not isinstance(weights,dict) or not all(isinstance(v,torch.Tensor) for v in weights.values()):findings.append('Weight file contains non-tensor metadata')
    manifest=json.loads((ROOT/'weights/model.json').read_text(encoding='utf8'))
    if hashlib.sha256((ROOT/'weights/lanwind.pt').read_bytes()).hexdigest()!=manifest['sha256']:findings.append('Checkpoint digest mismatch')
    if findings:raise SystemExit('\n'.join(findings))
    print(f'RELEASE_AUDIT_PASSED: {len(files)} public files; no data containers, private context or credentials; bilingual links valid; tensor-only weights verified')

if __name__=='__main__':main()
