from elastograde.data.audit import audit
from elastograde.reporting.provenance import digest

def clinical_readiness(manifest,checkpoint=None):
    reasons=[]
    for key in ['organ','grading_system','pathology_reference','acquisition_protocol','ethics_provenance','label_definition']:
        if not manifest.get(key):reasons.append('Missing '+key)
    records=manifest.get('records',[])
    if not records:reasons.append('No pathology-linked records')
    for r in records:
        for key in ['patient_id','lesion_id','pathology_grade','pathology_date','imaging_date','linkage_verified','label_provenance']:
            if r.get(key) is None or r.get(key) is False:reasons.append('Incomplete record '+key);break
        if r.get('data_origin')=='synthetic_fixture':reasons.append('Synthetic labels cannot establish clinical readiness')
    if records:
        try:audit(records,'patient_id')
        except ValueError as e:reasons.append(str(e))
        expected=set(manifest.get('classes',[]))
        minimum=manifest.get('minimum_patients_per_class_per_role')
        if not expected or not isinstance(minimum,int) or minimum<2:reasons.append('Predeclared independent-patient minimum and classes required')
        else:
            for role in ['train','validation','calibration','test']:
                for grade in expected:
                    n=len({r.get('patient_id') for r in records if r.get('split')==role and r.get('pathology_grade')==grade})
                    if n<minimum:reasons.append(f'Insufficient {role} patients for grade {grade}: {n}')
    if checkpoint is None:reasons.append('No compatible calibrated checkpoint metadata')
    else:
        for key in ['grading_system','organ']:
            if checkpoint.get(key)!=manifest.get(key):reasons.append('Checkpoint mismatch: '+key)
        if checkpoint.get('manifest_hash')!=digest(manifest):reasons.append('Checkpoint manifest hash mismatch')
        if not checkpoint.get('calibration_run_id') or not checkpoint.get('validation_report_sha256'):reasons.append('Missing calibration/validation provenance')
    return {'eligible':not reasons,'grade_status':'research_eligible' if not reasons else 'blocked','grade_probabilities':None,'reasons':sorted(set(reasons)),'clinical_validation':'not_established_by_software_gate'}
