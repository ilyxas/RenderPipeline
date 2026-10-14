"""Profile-requested real-time material conversions for the GLB export session (Blender only).

glTF has no procedural nodes: what the asset builds procedurally (lace crop/ramp alpha,
knit bump) is baked into textures here, and fabric is tuned for real-time viewers.
Everything runs in the export session only; the canonical asset is never saved.
Each function returns records of what it changed, for animation.export.json.
"""
import zlib
import numpy as np


def _bsdf(material):
    return next(n for n in material.node_tree.nodes if n.type=='BSDF_PRINCIPLED')


def _strip(material):
    nt=material.node_tree;bsdf=_bsdf(material)
    for n in [n for n in nt.nodes if n.name!=bsdf.name and n.type!='OUTPUT_MATERIAL']:nt.nodes.remove(n)
    return bsdf


def _srgb(c):
    return c*12.92 if c<=.0031308 else 1.055*c**(1/2.4)-.055


def lace_alpha(materials,images,spec):
    """Lace image cropped by its MULTIPLY_ADD span, luminance ramp -> alpha**gamma, flat dark base."""
    mat=materials[spec['material']];nt=mat.node_tree;size=spec.get('size',1024)
    tex=next(n for n in nt.nodes if n.type=='TEX_IMAGE');ramp=next(n for n in nt.nodes if n.type=='VALTORGB').color_ramp
    crop=next(n for n in nt.nodes if n.type=='MATH' and n.operation=='MULTIPLY_ADD')
    cs,co=crop.inputs[1].default_value,crop.inputs[2].default_value;src=tex.image;w,h=src.size
    px=np.array(src.pixels[:],dtype=np.float32).reshape(h,w,4)
    xs=np.clip(((co+cs*(np.arange(size)+.5)/size)*w).astype(int),0,w-1);ys=np.clip(((np.arange(size)+.5)/size*h).astype(int),0,h-1)
    sub=px[ys][:,xs,:3];lin=np.where(sub<=.04045,sub/12.92,((sub+.055)/1.055)**2.4)
    lum=lin@np.array([.2126,.7152,.0722],dtype=np.float32)
    lo,hi=ramp.elements[0].position,ramp.elements[1].position
    alpha=np.clip((hi-lum)/(hi-lo),0,1)**spec.get('gamma',.6)
    out=np.empty((size,size,4),dtype=np.float32);out[...,:3]=[_srgb(c) for c in spec['base_color']];out[...,3]=alpha
    img=images.new(spec['material']+'_Alpha',size,size,alpha=True);img.pixels.foreach_set(out.ravel());img.file_format='PNG';img.pack()
    bsdf=_strip(mat);t=nt.nodes.new('ShaderNodeTexImage');t.image=img;t.extension='REPEAT'
    nt.links.new(t.outputs['Color'],bsdf.inputs['Base Color']);nt.links.new(t.outputs['Alpha'],bsdf.inputs['Alpha'])
    return [{'stage':'glb','kind':'conversion','material':spec['material'],'change':'procedural lace -> baked RGBA texture','size':size,'gamma':spec.get('gamma',.6),'mean_alpha':round(float(alpha.mean()),4)}]


def periodic_noise(size,cells,seed):
    """Tileable value noise: band-limited random field via FFT (wraps on both axes)."""
    f=np.fft.fft2(np.random.default_rng(seed).standard_normal((size,size)))
    k=np.sqrt(np.fft.fftfreq(size)[:,None]**2+np.fft.fftfreq(size)[None,:]**2)*size
    n=np.real(np.fft.ifft2(f*np.exp(-(k/cells)**2)));n-=n.mean()
    return n/(np.abs(n).max()+1e-9)


def knit_normal(size,ribs,tile_ribs,depth,knit,seed):
    """Height (knit noise + optional ribs) -> tangent-space normal pixels, seamless."""
    x=(np.arange(size)+.5)/size;h=knit*periodic_noise(size,90,seed)
    if ribs:h=h+depth*np.sin(2*np.pi*tile_ribs*x)[None,:]
    s=.9/size*40;dx=(np.roll(h,-1,1)-np.roll(h,1,1))*s;dy=(np.roll(h,-1,0)-np.roll(h,1,0))*s
    nrm=np.dstack([-dx,-dy,np.ones_like(h)]);nrm/=np.linalg.norm(nrm,axis=2,keepdims=True)
    px=np.ones((size,size,4),dtype=np.float32);px[...,:3]=nrm*.5+.5;return px


def normal_maps(materials,images,spec):
    """Procedural fabric bump -> tileable normal map on spec['uv_map'], scaled by a Mapping node (KHR_texture_transform)."""
    records=[];size=spec.get('tile',512)
    for name,(ribs,tile_ribs,tile_h,depth,knit) in spec['materials'].items():
        if name not in materials:records.append({'stage':'glb','kind':'conversion','material':name,'skipped':'material missing'});continue
        mat=materials[name];nt=mat.node_tree;bsdf=_strip(mat)
        img=images.new(name+'_Normal',size,size,alpha=False);img.colorspace_settings.name='Non-Color'
        img.pixels.foreach_set(knit_normal(size,ribs,tile_ribs,depth,knit,zlib.crc32(name.encode())&0xffff).ravel());img.file_format='PNG';img.pack()
        uv=nt.nodes.new('ShaderNodeUVMap');uv.uv_map=spec['uv_map'];mp=nt.nodes.new('ShaderNodeMapping');mp.vector_type='POINT'
        scale=((ribs/tile_ribs) if ribs else spec['plain_u_scale'],1/tile_h,1);mp.inputs['Scale'].default_value=scale
        tex=nt.nodes.new('ShaderNodeTexImage');tex.image=img;tex.extension='REPEAT';nm=nt.nodes.new('ShaderNodeNormalMap');nm.uv_map=spec['uv_map']
        nt.links.new(uv.outputs['UV'],mp.inputs['Vector']);nt.links.new(mp.outputs['Vector'],tex.inputs['Vector'])
        nt.links.new(tex.outputs['Color'],nm.inputs['Color']);nt.links.new(nm.outputs['Normal'],bsdf.inputs['Normal'])
        records.append({'stage':'glb','kind':'conversion','material':name,'change':'procedural knit -> baked normal map','uv_scale':[round(float(v),3) for v in scale]})
    return records


def realtime_fabric(materials,spec,set_input):
    """Matte black fabric for real-time viewers: no sheen, set specular/roughness, dark base where unlinked."""
    records=[]
    for m in materials:
        if not m.name.startswith(spec['prefix']) or not m.node_tree:continue
        b=_bsdf(m);c=spec['base']
        for socket,value in (('Sheen Weight',spec['sheen']),('Specular IOR Level',spec['specular']),('Roughness',spec['roughness'])):set_input(records,m.name,b,socket,value,stage='glb',kind='conversion')
        if not b.inputs['Base Color'].is_linked:set_input(records,m.name,b,'Base Color',(c,c,c*spec.get('blue_ratio',1.12),1),stage='glb',kind='conversion')
    return records


def hair_alpha_round(materials,spec):
    """Round the hair alpha so the glTF exporter writes alphaMode MASK (cutoff .5)."""
    mat=materials[spec['material']];nt=mat.node_tree;bsdf=_bsdf(mat)
    link=next(l for l in nt.links if l.to_node.name==bsdf.name and l.to_socket.name=='Alpha')
    rnd=nt.nodes.new('ShaderNodeMath');rnd.operation='ROUND';nt.links.new(link.from_socket,rnd.inputs[0]);nt.links.new(rnd.outputs[0],bsdf.inputs['Alpha'])
    return [{'stage':'glb','kind':'conversion','material':spec['material'],'input':'Alpha','change':'alpha -> ROUND (glTF alphaMode MASK)'}]


def apply(materials,images,glb,set_input):
    """Run the conversions a profile's look['glb'] requests, in the reference order."""
    records=[]
    if glb.get('lace_alpha'):records+=lace_alpha(materials,images,glb['lace_alpha'])
    if glb.get('hair_alpha'):records+=hair_alpha_round(materials,glb['hair_alpha'])
    if glb.get('normal_maps'):records+=normal_maps(materials,images,glb['normal_maps'])
    if glb.get('realtime_fabric'):records+=realtime_fabric(materials,glb['realtime_fabric'],set_input)
    return records
