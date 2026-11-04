"""Socket-free notebook verification in a fresh Python process per notebook.

Executes Python cells through IPython, captures streams/displays and figures.
This is an explicit fallback, not represented as a Jupyter-kernel execution.
"""
from pathlib import Path
import sys,io,base64,time,traceback
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from IPython.display import display,Image

def run(source,destination,root):
    import os
    os.chdir(root)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def show(*args,**kwargs):
        for number in plt.get_fignums():
            fig=plt.figure(number);buf=io.BytesIO();fig.savefig(buf,format='png',bbox_inches='tight',dpi=100)
            display(Image(data=buf.getvalue()));plt.close(fig)
    plt.show=show
    shell=InteractiveShell.instance();nb=nbformat.read(source,4);nbformat.validate(nb);count=0;failed=False
    for cell in nb.cells:
        if cell.cell_type!='code':continue
        count+=1;cell.execution_count=count;cell.outputs=[]
        with capture_output(stdout=True,stderr=True,display=True) as captured:
            result=shell.run_cell(cell.source,store_history=True)
        if captured.stdout:cell.outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
        if captured.stderr:cell.outputs.append(nbformat.v4.new_output('stream',name='stderr',text=captured.stderr))
        for output in captured.outputs:
            data={k:(base64.b64encode(v).decode() if isinstance(v,bytes) else v) for k,v in output.data.items()}
            cell.outputs.append(nbformat.v4.new_output('display_data',data=data,metadata=output.metadata))
        error=result.error_before_exec or result.error_in_exec
        if error:
            cell.outputs.append(nbformat.v4.new_output('error',ename=type(error).__name__,evalue=str(error),traceback=traceback.format_exception(type(error),error,error.__traceback__)));failed=True;break
    nb.metadata['execution_backend']='fresh_python_process_ipython_no_kernel_sockets'
    nbformat.write(nb,destination)
    return not failed
if __name__=='__main__':raise SystemExit(0 if run(*sys.argv[1:4]) else 1)
