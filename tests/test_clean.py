import json,tempfile,unittest
from pathlib import Path
from unittest import mock
from xms import clean


def make_run(root,status):
    run=root/status;(run/'frames').mkdir(parents=True);(run/'comparison/frames').mkdir(parents=True);(run/'outputs').mkdir()
    (run/'manifest.json').write_text(json.dumps({'schema_version':'xms.run.v1','status':status}))
    (run/'frames/manifest.json').write_text('{}');(run/'outputs/video.mp4').write_bytes(b'v')
    for i in range(3):(run/f'frames/{i:06d}.png').write_bytes(b'x'*10);(run/f'comparison/frames/{i:06d}.png').write_bytes(b'y'*5)
    return run


class Clean(unittest.TestCase):
    def test_prune_keeps_results_and_frame_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=make_run(Path(tmp),'rendered_pending_visual_review');self.assertEqual(clean.prune_frames(run),45)
            self.assertFalse(list(run.rglob('*.png')));self.assertTrue((run/'frames/manifest.json').exists());self.assertTrue((run/'outputs/video.mp4').exists())
    def test_dry_run_changes_nothing_and_running_runs_are_skipped_while_busy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);done=make_run(root,'cancelled');live=make_run(root,'running')
            with mock.patch.object(clean,'active_runs',return_value=True):
                self.assertEqual(clean.clean(root,log=lambda m:None),45);self.assertEqual(len(list(root.rglob('*.png'))),12)
                clean.clean(root,apply=True,delete=True,log=lambda m:None)
            self.assertFalse(list(done.rglob('*.png')));self.assertEqual(len(list(live.rglob('*.png'))),6)


if __name__=='__main__':unittest.main()
