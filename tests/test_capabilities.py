from elastograde.capabilities import clinical_readiness
from elastograde.inference.pipeline import infer
from elastograde.data.synthetic import make_pair

def test_clinical_gate():assert not clinical_readiness({'organ':'breast','grading_system':'not_verified'})['eligible']
def test_no_grade_default():
 r=infer(make_pair());assert r['grade_probabilities'] is None and r['data_origin']=='synthetic_fixture'
