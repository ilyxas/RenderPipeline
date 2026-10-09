import numpy as np


def measure(result):
    v=result['validity'];return {'status':'measured','channel_coverage':v.mean(axis=0).tolist(),'inferred_samples':int((result['provenance']==2).sum()),'event_quality_accepted':False,**result.get('diagnostics',{})}
