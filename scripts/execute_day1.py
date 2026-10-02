"""Execute both notebooks with this Python interpreter; fail on cell errors."""
from pathlib import Path
import os,sys,time,json,subprocess
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT=Path(__file__).resolve().parents[1]
os.environ['ESS_PROJECT_ROOT']=str(ROOT)
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'results/day1/.cache/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'results/day1/.cache'))
subprocess.run([sys.executable,str(ROOT/'scripts/build_notebooks.py')],check=True)
checks=[]
for fn in ['01_data_check.ipynb','02_day1_eda.ipynb']:
    path=ROOT/'notebooks'/fn;nb=nbformat.read(path,as_version=4)
    km=KernelManager(kernel_name='python3')
    km.kernel_spec.argv[0]=sys.executable
    start=time.monotonic();print('Executing',fn,flush=True)
    client=NotebookClient(nb,timeout=900,allow_errors=False,km=km,resources={'metadata':{'path':str(ROOT/'notebooks')}})
    try:
        client.execute()
    except Exception:
        nbformat.write(nb,path)
        raise
    finally:
        if km.has_kernel:
            km.shutdown_kernel(now=True)
        km.cleanup_resources()
    nbformat.write(nb,path)
    errors=[o for c in nb.cells for o in c.get('outputs',[]) if o.output_type=='error']
    assert not errors
    code_cells=[c for c in nb.cells if c.cell_type=='code']
    assert all(c.execution_count is not None for c in code_cells)
    checks.append({'notebook':fn,'code_cells':len(code_cells),'errors':len(errors),'seconds':round(time.monotonic()-start,2),'all_executed':True})
    print(checks[-1],flush=True)
    from nbconvert import HTMLExporter
    html,_=HTMLExporter().from_notebook_node(nb)
    (ROOT/'results/day1'/fn.replace('.ipynb','.html')).write_text(html,encoding='utf-8')
(ROOT/'results/day1/notebook_execution.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2))
subprocess.run([sys.executable,str(ROOT/'scripts/verify_day1.py')],check=True)
