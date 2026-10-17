import json,tempfile,unittest
from pathlib import Path
from unittest import mock
from xms import registry
from xms.animation.io import file_hash
from xms.profiles.character import profile_digest


def scene_profile(folder,digest,revision):
    folder.mkdir(parents=True);data={'schema_version':'xms.scene.v1','collection':'Loft','world':'W','asset_sha256':digest,'profile_revision':revision}
    data['profile_hash']=profile_digest(data);(folder/'scene.json').write_text(json.dumps(data));return folder/'scene.json'


class RegistryTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();root=Path(self.tmp.name);blender=root/'blender';blender.mkdir()
        self.old=blender/'Loft_v09.blend';self.old.write_bytes(b'old');self.new=blender/'Loft_v10.blend';self.new.write_bytes(b'new')
        char=blender/'Xandra.blend';char.write_bytes(b'char')
        cp=root/'profiles/characters/xandra/v1';cp.mkdir(parents=True);(cp/'rig.npz').write_bytes(b'rig')
        data={'schema_version':'xms.character.v1','asset_sha256':file_hash(char),'files':{'rig.npz':file_hash(cp/'rig.npz')},'profile_revision':1}
        data['profile_hash']=profile_digest(data);(cp/'character.json').write_text(json.dumps(data))
        self.v3=scene_profile(root/'profiles/scenes/loft/v3',file_hash(self.old),2);self.v4=scene_profile(root/'profiles/scenes/loft/v4',file_hash(self.new),3)
        (root/'.xms').mkdir();self.config=root/'.xms/config.json';self.blender=root/'blender-bin';self.blender.write_text('')
        # The 2026-10-17 breakage: profile loft/v3 paired with the Loft v10 file.
        self.config.write_text(json.dumps({'blender':str(self.blender),'character_asset':str(char),'character_profile':str(cp),'scene_asset':str(self.new),'scene_profile':str(self.v3)}))
        self.patches=[mock.patch.object(registry,'PROFILES',root/'profiles'),mock.patch.object(registry,'ASSET_DIRS',(blender,)),mock.patch.object(registry,'CACHE',root/'.xms/hash-cache.json')]
        for p in self.patches:p.start()

    def tearDown(self):
        for p in self.patches:p.stop()
        self.tmp.cleanup()

    def test_status_explains_mismatch_and_suggests_profile(self):
        s=registry.status(self.config)
        self.assertFalse(s['ok']);self.assertTrue(s['current']['character']['ok'])
        scene=s['current']['scene'];self.assertIn('loft/v4',scene['problem']);self.assertEqual(scene['suggest'],['loft/v4'])
        self.assertEqual(s['unregistered'],[])
        with self.assertRaisesRegex(ValueError,'подходит профиль loft/v4'):registry.check({'_config_path':str(self.config)})

    def test_use_writes_matching_pair_with_backup(self):
        s=registry.use('loft/v3',self.config)
        self.assertTrue(s['ok']);config=json.loads(self.config.read_text())
        self.assertEqual(config['scene_asset'],str(self.old));self.assertEqual(Path(config['scene_profile']),self.v3)
        self.assertEqual(len(list((self.config.parent/'backups').glob('config.*.json'))),1)

    def test_use_refuses_profile_without_file(self):
        self.old.unlink()
        with self.assertRaisesRegex(ValueError,'нет файла'):registry.use('loft/v3',self.config)

    def test_register_changed_file_as_next_version(self):
        self.new.write_bytes(b'edited in Blender')
        s=registry.status(self.config);self.assertFalse(s['current']['scene']['ok']);self.assertIn('зарегистрировать',s['current']['scene']['problem'])
        self.assertEqual([u['name'] for u in s['unregistered']],['Loft_v10.blend'])
        before=self.v4.read_text()
        with mock.patch.object(registry,'_blender_check',return_value={'ok':True}) as check:s=registry.register(self.new,'loft/v4',self.config)
        self.assertEqual(check.call_args.args[2],'scene');self.assertEqual(self.v4.read_text(),before)
        v5=json.loads((self.v4.parent.parent/'v5/scene.json').read_text())
        self.assertEqual(v5['asset_sha256'],file_hash(self.new));self.assertEqual(v5['profile_revision'],4);self.assertEqual(v5['profile_hash'],profile_digest(v5))
        self.assertTrue(s['ok']);self.assertEqual(s['current']['scene']['profile'],'loft/v5');self.assertEqual(s['unregistered'],[])

    def test_register_refuses_known_file_and_failed_check(self):
        with self.assertRaisesRegex(ValueError,'уже зарегистрирован как loft/v4'):registry.register(self.new,'loft/v3',self.config)
        self.new.write_bytes(b'other scene')
        with mock.patch.object(registry,'_blender_check',side_effect=ValueError('нет коллекции Loft')):
            with self.assertRaisesRegex(ValueError,'нет коллекции'):registry.register(self.new,'loft/v4',self.config)
        self.assertFalse((self.v4.parent.parent/'v5').exists())


if __name__=='__main__':unittest.main()
