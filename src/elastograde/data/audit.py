def audit(records,group_key='geometry_id'):
    owners={};ids=set()
    for r in records:
        if not r.get('sample_id') or r['sample_id'] in ids:raise ValueError('Missing/duplicate sample_id')
        ids.add(r['sample_id'])
        if r.get('split') not in ('train','validation','calibration','test'):raise ValueError('Invalid split role')
        if not r.get(group_key):raise ValueError('Missing verified group ID')
        keys=[(group_key,r[group_key])]
        for key in ['source_sha256','patient_id','geometry_id','acquisition_id','sequence_id']:
            if r.get(key):keys.append((key,r[key]))
        keys += [('frame',f) for f in r.get('frame_ids',[]) or []]
        keys += [('ancestor',f) for f in r.get('ancestor_ids',[]) or []]
        for key in keys:
            if key in owners and owners[key]!=r['split']:raise ValueError(f'Leakage detected: {key}')
            owners[key]=r['split']
    return {'status':'passed','samples':len(records),'groups':len(set(r[group_key] for r in records)),'group_key':group_key}
