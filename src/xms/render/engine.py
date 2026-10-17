"""Render engines: EEVEE (default, fast, for iteration) and Cycles (path traced, for finals).

The engine is a render setting only: the bundle, asset, scene and camera are the same for both.
"""
ENGINES=('eevee','cycles')
# Cycles: adaptive sampling + OpenImageDenoise on the GPU; motion blur over half a frame.
CYCLES={'preview':{'samples':32},'final':{'samples':64}}
CYCLES_COMMON={'adaptive_threshold':0.02,'max_bounces':8,'motion_blur_shutter':0.5}
# Rough seconds per frame per megapixel on an Apple M4 (measured 2026-10-14/16: EEVEE final 1080x1920 ~1.2 s,
# Cycles 64 samples 1080x1920 ~15 s; preview 32 samples 540x960 3.7 s, run 20261016-032906-988e05f7; persistent data cuts ~1.3 s/frame of reloads). Used only for the time estimate shown before a run.
_RATE={'eevee':{'preview':1.2,'final':0.6},'cycles':{'preview':3.6,'final':6.5}}
_OVERHEAD={'eevee':0.4,'cycles':0.2}
# Samples when none is chosen (quality default) and the choices offered in Studio; --samples accepts any 1..4096.
DEFAULT_SAMPLES={'eevee':{'preview':16,'final':32},'cycles':{'preview':CYCLES['preview']['samples'],'final':CYCLES['final']['samples']}}
SAMPLE_CHOICES={'eevee':(8,16,24,32,48,64,128),'cycles':(16,24,32,48,64,96,128,256)}
# Per-megapixel time grows like (samples + _SAMPLE_FIXED): part of the frame cost does not depend on samples.
_SAMPLE_FIXED={'eevee':4,'cycles':8}


def check(engine):
    if engine not in ENGINES:raise ValueError('Unknown render engine: '+str(engine))
    return engine


def exposure(scene_profile,engine):
    """Scene exposure for this engine: profiles may tune it per engine (EEVEE and Cycles light a room differently)."""
    return scene_profile.get('engine_exposure',{}).get(check(engine),scene_profile['exposure'])


def samples_for(engine,quality,samples=None):
    if samples is None:return DEFAULT_SAMPLES[check(engine)][quality]
    if isinstance(samples,bool) or not isinstance(samples,int) or not 1<=samples<=4096:raise ValueError('Samples must be an integer 1..4096: '+str(samples))
    return samples


def estimate_seconds(engine,quality,resolution,frames,samples=None):
    megapixels=resolution[0]*resolution[1]/1e6;base=DEFAULT_SAMPLES[check(engine)][quality];fixed=_SAMPLE_FIXED[engine]
    return frames*(_RATE[engine][quality]*(samples_for(engine,quality,samples)+fixed)/(base+fixed)*megapixels+_OVERHEAD[engine])


def configure(scene,engine,quality,preferences=None,samples=None):
    """Switch a Blender scene to the engine; return the settings recorded in the render manifest."""
    samples=samples_for(engine,quality,samples)
    if engine=='eevee':
        scene.eevee.taa_render_samples=samples;scene.render.engine='BLENDER_EEVEE_NEXT';scene.render.use_motion_blur=False;scene.render.use_persistent_data=False
        return {'engine':'eevee','samples':scene.eevee.taa_render_samples}
    c=scene.cycles;scene.render.engine='CYCLES';c.device='GPU';backend=None
    if preferences is not None:
        for name in ('METAL','OPTIX','CUDA','HIP','ONEAPI'):
            try:preferences.compute_device_type=name
            except TypeError:continue
            preferences.get_devices()
            if any(d.type==name for d in preferences.devices):
                for d in preferences.devices:d.use=d.type==name
                backend=name;break
    if backend is None:c.device='CPU'
    c.samples=samples;c.use_adaptive_sampling=True;c.adaptive_threshold=CYCLES_COMMON['adaptive_threshold']
    c.use_denoising=True;c.denoiser='OPENIMAGEDENOISE';c.denoising_use_gpu=backend is not None
    c.max_bounces=CYCLES_COMMON['max_bounces'];c.caustics_reflective=False;c.caustics_refractive=False;c.film_transparent=False
    scene.render.use_motion_blur=True;scene.render.motion_blur_shutter=CYCLES_COMMON['motion_blur_shutter']
    # Keep textures and static geometry on the device between frames: 3.4 -> 2.0 s/frame, same pixels (2026-10-16).
    scene.render.use_persistent_data=True
    return {'engine':'cycles','persistent_data':True,'samples':c.samples,'device':backend or 'CPU','denoiser':'OPENIMAGEDENOISE','adaptive_threshold':c.adaptive_threshold,'motion_blur_shutter':scene.render.motion_blur_shutter}
