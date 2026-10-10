"""Cheap catastrophic-motion gate, independent of frozen benchmark acceptance."""
from xms.qa.continuity import measure


def reasons(bundle,surface=None):
    failures=[];continuity=measure(bundle)
    if continuity['max_rotation_step_deg']>90:
        failures.append(f"Adjacent rotation jump {continuity['max_rotation_step_deg']:.1f} degrees exceeds 90 degrees")
    proxy=bundle.metadata.get('collision_proxy_diagnostics',{})
    if proxy.get('max_proxy_depth_m',0)>.06:
        failures.append(f"Forearm/torso proxy overlap {proxy['max_proxy_depth_m']*100:.1f} cm exceeds 6 cm")
    if surface and surface.get('signed_interior_available') and surface.get('max_candidate_depth_m',0)>.02:
        failures.append(f"Sampled signed garment depth {surface['max_candidate_depth_m']*100:.1f} cm exceeds 2 cm; inspect surface evidence")
    return failures
