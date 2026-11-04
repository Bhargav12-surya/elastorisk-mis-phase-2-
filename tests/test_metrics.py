import numpy as np,pytest
from elastograde.evaluation.metrics import displacement_metrics,segmentation_metrics
from elastograde.evaluation.grouped_statistics import grouped_bootstrap
from elastograde.uncertainty.calibration import TemperatureScaler,conformal_radius
from elastograde.uncertainty.selective import risk_coverage

def test_perfect_displacement():
 u=np.zeros((2,8,8));assert displacement_metrics(u,u,np.ones((8,8),bool))['endpoint_error']==0
def test_empty_support():
 with pytest.raises(ValueError):displacement_metrics(np.zeros((2,8,8)),np.zeros((2,8,8)),np.zeros((8,8),bool))
def test_group_bootstrap():
 r=grouped_bootstrap([1,1,3,3],['a','a','b','b']);assert r['mean']==2 and r['independent_groups']==2
def test_calibration_role():
 with pytest.raises(ValueError):TemperatureScaler().fit([[1,2]],[1],['test'])
def test_conformal_small_n():assert np.isinf(conformal_radius([1,2],.1))
def test_risk_coverage():
 r=risk_coverage([1,2,3],[.1,.2,.3]);np.testing.assert_allclose(r['risk'],[1,1.5,2])
def test_dice():assert segmentation_metrics([1,0],[1,0])['dice']==1
