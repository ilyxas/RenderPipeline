"""Render engines: EEVEE (default, fast, for iteration) and Cycles (path traced, for finals).

The engine is a render setting only: the bundle, asset, scene and camera are the same for both.
"""
ENGINES=('eevee','cycles')
# Cycles: adaptive sampling + OpenImageDenoise on the GPU; motion blur over half a frame.
CYCLES={'preview':{'samples':32},'final':{'samples':64}}
CYCLES_COMMON={'adaptive_threshold':0.02,'max_bounces':8,'motion_blur_shutter':0.5}
# Rough seconds per frame per megapixel on an Apple M4 (measured 2026-10-14/16: EEVEE final 1080x1920 ~1.2 s,
# Cycles 64 samples 1080x1920 ~15 s; preview 32 samples 540x960 3.7 s, run 20261016-032906-988e05f7). Used only for the time estimate shown before a run.
_RATE={'eevee':{'preview':1.2,'final':0.6},'cycles':{'preview':5.0,'final':7.2}}
_OVERHEAD={'eevee':0.4,'cycles':1.2}


def check(engine):
    if engine not in ENGINES:raise ValueError('Unknown render engine: '+str(engine))
    return engine


def exposure(scene_profile,engine):
    """Scene exposure for this engine: profiles may tune it per engine (EEVEE and Cycles light a room differently)."""
    return scene_profile.get('engine_exposure',{}).get(check(engine),scene_profile['exposure'])


def estimate_seconds(engine,quality,resolution,frames):
    megapixels=resolution[0]*resolution[1]/1e6
    return frames*(_RATE[check(engine)][quality]*megapixels+_OVERHEAD[engine])


def configure(scene,engine,quality,preferences=None):
    """Switch a Blender scene to the engine; return the settings recorded in the render manifest."""
    if check(engine)=='eevee':
        scene.render.engine='BLENDER_EEVEE_NEXT';scene.render.use_motion_blur=False
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
    c.samples=CYCLES[quality]['samples'];c.use_adaptive_sampling=True;c.adaptive_threshold=CYCLES_COMMON['adaptive_threshold']
    c.use_denoising=True;c.denoiser='OPENIMAGEDENOISE';c.denoising_use_gpu=backend is not None
    c.max_bounces=CYCLES_COMMON['max_bounces'];c.caustics_reflective=False;c.caustics_refractive=False;c.film_transparent=False
    scene.render.use_motion_blur=True;scene.render.motion_blur_shutter=CYCLES_COMMON['motion_blur_shutter']
    return {'engine':'cycles','samples':c.samples,'device':backend or 'CPU','denoiser':'OPENIMAGEDENOISE','adaptive_threshold':c.adaptive_threshold,'motion_blur_shutter':scene.render.motion_blur_shutter}
