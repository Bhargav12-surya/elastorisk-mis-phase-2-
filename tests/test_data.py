import numpy as np,pytest,zipfile
from elastograde.data.schemas import save_sample,load_sample
from elastograde.data.synthetic import make_pair
from elastograde.data.acquisition import extract_archive
from elastograde.data.impact_acquired import prepare

def test_roundtrip(tmp_path):
 p=save_sample(tmp_path/'sample.npz',make_pair());s=load_sample(p);assert s.metadata['data_origin']=='synthetic_fixture'
def test_corruption_checksum(tmp_path):
 p=save_sample(tmp_path/'sample.npz',make_pair());p.write_bytes(p.read_bytes()+b'bad')
 with pytest.raises(ValueError,match='checksum'):load_sample(p)
@pytest.mark.parametrize('kind',['constant','nan'])
def test_invalid_input(kind):
 s=make_pair();s.pre[:]=0 if kind=='constant' else np.nan
 with pytest.raises(ValueError):s.validate()
def test_zip_traversal(tmp_path):
 p=tmp_path/'a.zip'
 with zipfile.ZipFile(p,'w') as z:z.writestr('../escape.txt','oops')
 with pytest.raises(ValueError,match='Unsafe'):extract_archive(p,tmp_path/'out')
def test_multipart_missing(tmp_path):
 p=tmp_path/'data.zip.001';p.write_bytes(b'x')
 with pytest.raises(ValueError,match='Missing'):extract_archive(p,tmp_path/'out',2)
def test_explicit_adapter(tmp_path):
 s=make_pair();np.savez(tmp_path/'raw.npz',a=s.pre,b=s.post)
 c={'root':str(tmp_path),'output':str(tmp_path/'processed'),'dataset_id':'test','axes':'zx','mapping':{'pre':'a','post':'b'},'records':[{'sample_id':'a','path':'raw.npz','acquisition_id':'acq1'}]}
 r=prepare(c);assert load_sample(r[0]['path']).pre.shape==s.pre.shape
