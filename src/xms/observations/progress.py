"""Small stdout progress records; observations and tracker settings are unchanged."""
def report(stage,completed,total):
    if completed==total or completed%max(1,total//100)==0:
        print(f'XMS_PROGRESS {stage} {completed} {total}',flush=True)
