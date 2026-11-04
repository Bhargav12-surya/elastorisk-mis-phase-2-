import pytest
from elastograde.data.splits import grouped_split
from elastograde.data.audit import audit

def records():return [{'sample_id':str(i),'geometry_id':str(i//2),'frame_ids':[str(i)]} for i in range(20)]
def test_grouped():
 r=grouped_split(records());assert audit(r)['groups']==10
 for g in set(x['geometry_id'] for x in r):assert len(set(x['split'] for x in r if x['geometry_id']==g))==1
def test_frame_leak():
 r=grouped_split(records());a=r[0];b=next(x for x in r if x['split']!=a['split']);b['frame_ids']=a['frame_ids']
 with pytest.raises(ValueError,match='Leakage'):audit(r)
def test_missing_group():
 with pytest.raises(ValueError):grouped_split([{'sample_id':'x'}])
def test_no_test_fit():
 import numpy as np
 from elastograde.models.representations import MechanicalPhenotypes
 with pytest.raises(ValueError):MechanicalPhenotypes().fit(np.ones((10,2)),['test']*10)
