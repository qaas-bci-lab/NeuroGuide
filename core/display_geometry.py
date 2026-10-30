"""Native scan display coordinates; no display resampling or template rotation."""
from types import SimpleNamespace
import numpy as np


class DisplayGrid:
    def __init__(self, data, affine):
        self.mode = "original"
        self.data = data
        self.shape = data.shape
        self.affine = np.asarray(affine, dtype=float).copy()
        self.native_to_view = np.eye(4)

    def overlay_data(self, overlay):
        return overlay["data"]

    def voxel_to_native_ras(self, voxel):
        return (self.affine @ np.r_[voxel, 1])[:3]

    def native_ras_to_voxel(self, ras):
        return (np.linalg.inv(self.affine) @ np.r_[ras, 1])[:3]

    def native_voxel_to_view(self, voxel):
        return np.asarray(voxel, dtype=float)


def report_display_engine(engine):
    """Use the same native grid for report slices and viewer coordinates."""
    view = getattr(engine, "view", None)
    if view is None:
        return engine
    return SimpleNamespace(data=view.data, affine=view.affine, overlays=engine.overlays,
                           display_min=engine.display_min, display_max=engine.display_max,
                           show_locator=getattr(engine, "show_locator", True),
                           show_roi=getattr(engine, "show_roi", True),
                           clip_fields_to_roi=getattr(engine, "clip_fields_to_roi", False))
