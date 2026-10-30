"""Direct slice colors shared by anatomical and field 2.5D planes."""
import numpy as np
from matplotlib import colormaps


def slice_rgba(base,overlays,vmin,vmax):
    base=np.asarray(base,dtype=float)
    span=max(float(vmax)-float(vmin),1e-12)
    grey=np.nan_to_num(np.clip((base-vmin)/span,0,1),nan=0.)
    rgb=np.repeat(grey[...,None],3,axis=-1)
    visible=np.isfinite(base) & (base>vmin)
    for overlay in overlays or []:
        values=np.asarray(overlay['data']);valid=np.isfinite(values)
        if overlay.get('is_roi'):
            from scipy.ndimage import binary_erosion
            region=valid & (values>0)
            valid=region & ~binary_erosion(region)
            colors=np.broadcast_to(np.array([1.,.827,.545]),rgb.shape)
        else:
            lower=float(overlay['min']);upper=float(overlay['max'])
            valid &= values!=0 if overlay.get('is_atlas') else values>lower
            scaled=np.clip((values-lower)/max(upper-lower,1e-12),0,1)
            colors=colormaps[overlay.get('cmap','jet')](np.nan_to_num(scaled))[...,:3]
        alpha=1. if overlay.get('is_roi') else float(overlay.get('alpha',1.))
        rgb[valid]=(1-alpha)*rgb[valid]+alpha*colors[valid]
        if alpha>0:visible |= valid
    rgba=np.empty(base.shape+(4,),dtype=np.uint8)
    rgba[...,:3]=np.round(np.clip(rgb,0,1)*255).astype(np.uint8);rgba[...,3]=visible.astype(np.uint8)*255
    return rgba
