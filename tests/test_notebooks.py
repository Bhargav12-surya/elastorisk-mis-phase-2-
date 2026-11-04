from pathlib import Path
import nbformat

def test_notebook_contract():
 paths=list(Path('notebooks').glob('*.ipynb'));assert len(paths)==19
 for p in paths:
  n=nbformat.read(p,4);nbformat.validate(n);assert len(n.cells)>=6;assert any(c.cell_type=='code' for c in n.cells)
