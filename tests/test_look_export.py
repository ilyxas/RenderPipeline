"""Look overrides, kept outfit drivers and GLB skin accounting, without Blender (fake bpy data)."""
import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from xms.render.blender_adapter import apply_look,clear_legacy,driver_inputs,look_mode
from xms.export.skin_stats import masked,influences
from xms.render.glb_look import knit_normal

PROFILES=Path(__file__).resolve().parents[1]/'profiles/characters/xandra'


class Socket:
    def __init__(self,value,linked=False):self.default_value=value;self.is_linked=linked;self.links=[NS(from_socket='texture')] if linked else []


class Nodes(dict):
    def new(self,kind):
        node=NS(name=None,blend_type=None,inputs=[Socket(0),Socket(None),Socket(None)],outputs=[Socket(None)]);self['__new__']=node;return node


def material(spec,rough,linked_rough=False):
    bsdf=NS(name='Principled_glTF',inputs={'Base Color':Socket(None,True),'Specular IOR Level':Socket(spec),'Roughness':Socket(rough,linked_rough)})
    return NS(node_tree=NS(nodes=Nodes(Principled_glTF=bsdf),links=NS(remove=lambda l:None,new=lambda a,b:None)))


def materials():
    return {'MI_X3D_Hair_V7':material(.12,.65),'MI_X3D_Skin_Face_F4':material(.3,.5),'MI_X3D_Skin_Body_F4':material(.3,.5,linked_rough=True)}


class LookOverrides(unittest.TestCase):
    def test_legacy_profile_records_every_change(self):
        mats=materials();overrides=apply_look(PROFILES/'v6',mats)
        hair=[r for r in overrides if r['material']=='MI_X3D_Hair_V7']
        self.assertEqual(len(hair),1);self.assertIn('0.1697',hair[0]['after'])
        spec=[r for r in overrides if r['input']=='Specular IOR Level']
        self.assertTrue(spec and all(r['before']==.3 and r['after']==.32 for r in spec))
        self.assertEqual(mats['MI_X3D_Skin_Face_F4'].node_tree.nodes['Principled_glTF'].inputs['Specular IOR Level'].default_value,.32)
        linked=[r for r in overrides if r['material']=='MI_X3D_Skin_Body_F4' and r['input']=='Roughness'][0]
        self.assertEqual(linked['ignored'],'linked');self.assertEqual(mats['MI_X3D_Skin_Body_F4'].node_tree.nodes['Principled_glTF'].inputs['Roughness'].default_value,.5)
        self.assertTrue(all(r['kind']=='override' and r['stage']=='render' for r in overrides))

    def test_asset_mode_changes_nothing(self):
        mats=materials();self.assertEqual(apply_look(PROFILES/'v7',mats),[])
        self.assertEqual(mats['MI_X3D_Skin_Face_F4'].node_tree.nodes['Principled_glTF'].inputs['Specular IOR Level'].default_value,.3)
        self.assertNotIn('__new__',mats['MI_X3D_Hair_V7'].node_tree.nodes)

    def test_unknown_mode_is_refused(self):
        self.assertRaises(ValueError,look_mode,{'mode':'guess'})
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'look.json').write_text(json.dumps({'mode':'asset'}));self.assertEqual(apply_look(tmp,{}),[])


class Collection(list):
    def remove(self,item):list.remove(self,item)


def driver(path,target=None):
    variables=[NS(targets=[NS(id=target,data_path='["Sport_Set"]')])] if target else []
    return NS(data_path=path,driver=NS(variables=variables))


class KeptDrivers(unittest.TestCase):
    def setUp(self):
        self.root=NS(name='root',path_resolve=lambda path:True)
        self.garment=NS(type='MESH',data=NS(shape_keys=None),animation_data=NS(action='Legacy',nla_tracks=Collection(['track']),drivers=Collection([driver('hide_render',self.root),driver('hide_viewport',self.root),driver('location')])))
        keys=NS(animation_data=NS(action='Face',nla_tracks=Collection(),drivers=Collection([driver('key_blocks["jawOpen"].value')])))
        self.body=NS(type='MESH',data=NS(shape_keys=keys),animation_data=NS(action=None,nla_tracks=Collection(),drivers=Collection([driver('modifiers["Sport_Covered_Skin"].show_render',self.root)])))
        self.camera=NS(type='CAMERA',animation_data=None)

    def test_visibility_and_mask_drivers_survive_legacy_clear(self):
        record=clear_legacy([self.garment,self.body,self.camera])
        self.assertEqual([d.data_path for d in self.garment.animation_data.drivers],['hide_render','hide_viewport'])
        self.assertEqual([d.data_path for d in self.body.animation_data.drivers],['modifiers["Sport_Covered_Skin"].show_render'])
        self.assertEqual(record['kept_drivers'],3);self.assertEqual(record['removed_drivers'],2)

    def test_actions_nla_and_shape_key_drivers_are_removed(self):
        clear_legacy([self.garment,self.body])
        self.assertIsNone(self.garment.animation_data.action);self.assertEqual(len(self.garment.animation_data.nla_tracks),0)
        keys=self.body.data.shape_keys.animation_data;self.assertIsNone(keys.action);self.assertEqual(len(keys.drivers),0)

    def test_driver_inputs_name_the_outfit_switch(self):
        self.assertEqual(driver_inputs([self.garment,self.body,self.camera]),{'root["Sport_Set"]':True})


class SkinAccounting(unittest.TestCase):
    def test_mask_matches_blender_vertex_group_mode(self):
        w=[0,0.2,1]
        self.assertEqual(masked(w,0,invert=True).tolist(),[False,True,True])
        self.assertEqual(masked(w,0,invert=False).tolist(),[True,False,False])
        self.assertEqual(masked(w,.5,invert=True).tolist(),[False,False,True])

    def test_influence_counts_and_truncation(self):
        weights=[[.5,.5],[.2]*5,[.1]*12,[.3,.00005]]
        four=influences(weights,4);eight=influences(weights,8)
        self.assertEqual(four['max_influences_in_asset'],12);self.assertEqual(four['exported_influences'],4)
        self.assertEqual(four['truncated_vertices'],2);self.assertEqual(eight['truncated_vertices'],1)
        self.assertAlmostEqual(eight['max_truncated_weight'],.4)
        self.assertEqual(influences([[.3,.00005]],4)['max_influences_in_asset'],1)

    def test_knit_normal_is_deterministic_and_unit(self):
        a=knit_normal(32,420,8,1.,.45,7);b=knit_normal(32,420,8,1.,.45,7)
        self.assertTrue(np.array_equal(a,b))
        n=a[...,:3]*2-1;self.assertTrue(np.allclose(np.linalg.norm(n,axis=2),1,atol=1e-5))


class ReportBanner(unittest.TestCase):
    def test_banner_only_when_the_look_was_overridden(self):
        from xms.report.html import report
        base={'status':'needs_review','metrics':{},'warnings':['w1']}
        with tempfile.TemporaryDirectory() as tmp:
            report(Path(tmp)/'a',dict(base,look_overrides=[{'stage':'render','kind':'override','material':'Hair','input':'Base Color','before':'texture','after':'texture x tint'}]))
            report(Path(tmp)/'b',dict(base,look_overrides=[{'stage':'glb','kind':'conversion','material':'Lace','change':'baked'}]))
            a=(Path(tmp)/'a/report.html').read_text();b=(Path(tmp)/'b/report.html').read_text()
        self.assertIn('<div class="alert">',a);self.assertIn('Hair',a);self.assertNotIn('<div class="alert">',b);self.assertIn('w1',b)


class ProfileV7(unittest.TestCase):
    def test_v7_asks_for_eight_influences_and_no_render_overrides(self):
        look=json.loads((PROFILES/'v7/look.json').read_text())
        self.assertEqual(look['mode'],'asset');self.assertNotIn('hair',look);self.assertNotIn('skin',look)
        self.assertEqual(look['glb']['influences'],8)

    def test_older_profiles_keep_their_legacy_look(self):
        for v in ('v3','v4','v5','v6'):
            look=json.loads((PROFILES/v/'look.json').read_text());self.assertEqual(look_mode(look),'profile');self.assertIn('tint',look['hair'])


if __name__=='__main__':unittest.main()
