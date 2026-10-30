# ui/canvas.py

import os
import numpy as np
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import pyqtSignal, Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
os.environ.setdefault("QT_API", "pyqt6")

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    PYVISTA_AVAILABLE = True
except Exception:
    pv = None
    QtInteractor = None
    PYVISTA_AVAILABLE = False


class MriCanvas(FigureCanvas):
    clicked = pyqtSignal(str, float, float)
    drag_moved = pyqtSignal(str, float, float)
    drag_released = pyqtSignal(str, float, float)
    slice_scrolled = pyqtSignal(str, int)

    def __init__(self, title="", axis_type="axial"):
        self.fig = Figure(facecolor="black", constrained_layout=False)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.axis("off")
        self.title = title
        self.axis_type = axis_type
        self.cbars = []
        self._dragging = False
        self._last_event_time = 0
        self._scroll_remainder = 0.0
        self.fig.subplots_adjust(left=0.02, right=0.90, top=0.90, bottom=0.02)
        self.mpl_connect("button_press_event", self.on_mouse_press)
        self.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.mpl_connect("button_release_event", self.on_mouse_release)
        self.mpl_connect("scroll_event", self.on_mouse_scroll)

    def on_mouse_scroll(self, event):
        if event.inaxes is not self.ax:
            return
        step = float(getattr(event, 'step', 0))
        if not np.isfinite(step):
            return
        self._scroll_remainder += step
        slices = int(self._scroll_remainder)
        if slices:
            self._scroll_remainder -= slices
            self.slice_scrolled.emit(self.axis_type, slices)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Reserve a fixed logical pixel margin for the plane title even when
        # the new per-plane toolbar leaves a short canvas in a small window.
        height=max(1,self.height())
        self.fig.subplots_adjust(top=1-min(.35,38/height))
        self.draw_idle()

    def _emit_if_valid(self, signal, event):
        if event.inaxes is self.ax and event.xdata is not None and event.ydata is not None:
            signal.emit(self.axis_type, event.xdata, event.ydata)

    def on_mouse_press(self, event):
        if event.button == 1:
            self._dragging = True
            self._emit_if_valid(self.clicked, event)

    def on_mouse_move(self, event):
        if not self._dragging:
            return
        import time
        now = time.time() * 1000
        if now - self._last_event_time < 50:
            return
        self._last_event_time = now
        self._emit_if_valid(self.drag_moved, event)

    def on_mouse_release(self, event):
        if self._dragging:
            self._dragging = False
            self._emit_if_valid(self.drag_released, event)

    def _format_slice_for_display(self, data_slice):
        return np.flipud(np.rot90(data_slice))

    def render_slice(self, data_slice, overlays=None, vmin=None, vmax=None,
                     crosshair=None, aspect=1.0):
        self.ax.clear()
        self.ax.axis("off")
        for cb in self.cbars:
            cb.ax.remove()
        self.cbars = []
        self.ax.set_title(self.title, color="#bac3d2", fontsize=11,
                          fontweight="normal", pad=10)
        if data_slice is None:
            self._scroll_remainder = 0.0
            self.draw_idle()
            return
        img_show = self._format_slice_for_display(data_slice)
        self.ax.imshow(img_show, cmap="gray", vmin=vmin, vmax=vmax,
                       origin="lower", interpolation="nearest", aspect=aspect)
        if overlays:
            for i, ov in enumerate(overlays):
                ov_raw = ov["data"]
                if ov_raw is None:
                    continue
                ov_show = self._format_slice_for_display(ov_raw)
                if ov.get('is_roi'):
                    region=np.isfinite(ov_show) & (ov_show>0)
                    if np.any(region) and not np.all(region):
                        self.ax.contour(region.astype(float),levels=[.5],colors=['#ffd38b'],linewidths=1.0,zorder=8)
                    continue
                if ov.get("is_atlas"):
                    masked_data = np.ma.masked_equal(ov_show, 0)
                    interp = "nearest"
                else:
                    masked_data = np.ma.masked_invalid(ov_show)
                    masked_data = np.ma.masked_where(
                        masked_data <= ov["min"], masked_data)
                    interp = "bilinear"
                im = self.ax.imshow(
                    masked_data, cmap=ov["cmap"], alpha=ov["alpha"],
                    vmin=ov["min"], vmax=ov["max"], origin="lower",
                    interpolation=interp, aspect=aspect)
                if not ov.get("is_atlas"):
                    self.add_parallel_colorbar(im, i, ov.get('units','V/m'))
        if crosshair:
            cx, cy = crosshair
            self.ax.axhline(cy, color="#8195b8", linewidth=0.8,
                            linestyle="--", alpha=0.6)
            self.ax.axvline(cx, color="#8195b8", linewidth=0.8,
                            linestyle="--", alpha=0.6)
            self.ax.scatter([cx], [cy], s=15, c="#d0daeb",
                            edgecolors="black", linewidths=0.5, zorder=10)
        self.draw_idle()

    def add_parallel_colorbar(self, mappable, index, units='V/m'):
        ax_ins = inset_axes(
            self.ax, width="5%", height="60%", loc="lower left",
            bbox_to_anchor=(1.05 + index * 0.12, 0.2, 0.1, 0.6),
            bbox_transform=self.ax.transAxes, borderpad=0)
        cb = self.fig.colorbar(mappable, cax=ax_ins)
        cb.ax.yaxis.set_tick_params(color="white", labelcolor="white",
                                    labelsize=7)
        cb.outline.set_edgecolor("white")
        cb.ax.set_title(units,color='#bac3d2',fontsize=7,pad=4)
        self.cbars.append(cb)


# ---------- 3D Freeview 风格交互视图 ----------
class Freeview3DWidget(QWidget):
    slice_changed = pyqtSignal(int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.available = PYVISTA_AVAILABLE
        self._plotter = None
        self._first_render = True
        self._camera = None
        self._slice = {"x": 0, "y": 0, "z": 0}
        self._vol = self._ov = self._vmin = self._vmax = None
        self._affine=np.eye(4); self._inverse_affine=np.eye(4)
        self._step = 2
        self._pw=[]; self._ov_grids=[]; self._rebuilding=False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if not self.available:
            msg = QLabel("PyVista 3D 不可用\npip install pyvista pyvistaqt vtk")
            msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
            msg.setStyleSheet("color: white;")
            layout.addWidget(msg)
            return
        self._plotter = QtInteractor(self)
        self._plotter.set_background("black")
        self._plotter.hide_axes()
        layout.addWidget(self._plotter.interactor)

        # 中键平移视角
        try:
            style = self._plotter.iren.GetInteractorStyle()
            if style:
                # 确保中键响应平移
                style.SetMiddleButtonPan(True)
        except Exception:
            pass

    # ── 公共接口 ──
    def mark_rebuild_needed(self):
        self._need_full_rebuild = True

    def clear_volume(self):
        self._rebuilding=True
        had_volume=self._vol is not None
        if self._plotter is not None:
            self._plotter.clear_plane_widgets()
            self._plotter.clear()
            if had_volume: self._plotter.render()
        self._vol=self._ov=None; self._camera=None
        self._first_render=True; self._need_full_rebuild=True
        self._pw=[]; self._ov_grids=[]; self._rebuilding=False

    def shutdown(self):
        self._rebuilding=True; self._vol=self._ov=None
        if self._plotter is not None:
            self._plotter.clear_plane_widgets(); self._plotter.close(); self._plotter=None
        self._pw=[]; self._ov_grids=[]; self.available=False

    def render_3d(self, volume, x, y, z, overlays=None, vmin=None, vmax=None, affine=None):
        if not self.available or volume is None:
            return
        # Render pre-colored structured planes; no VTK overlay slice/threshold filters.
        x = int(np.clip(x, 0, volume.shape[0] - 1))
        y = int(np.clip(y, 0, volume.shape[1] - 1))
        z = int(np.clip(z, 0, volume.shape[2] - 1))
        affine=np.eye(4) if affine is None else np.asarray(affine,dtype=float)
        if affine.shape!=(4,4) or not np.isfinite(affine).all():
            raise ValueError("Invalid 3D volume affine")
        inverse=np.linalg.inv(affine)
        new_data = volume is not self._vol or not np.allclose(affine,self._affine,atol=1e-8,rtol=0)
        self._affine=affine.copy(); self._inverse_affine=inverse
        self._vol, self._ov = volume, overlays
        self._vmin, self._vmax = vmin, vmax
        self._slice = {"x": x, "y": y, "z": z}

        if new_data or getattr(self, "_need_full_rebuild", True):
            self._full_rebuild()
        else:
            self._refresh_slices_only()

    def reset_view(self, *_signal_args):
        """Reset only the camera; native slice positions and data stay intact."""
        if not self.available or self._plotter is None or self._vol is None:
            return
        from ui.volume_geometry import volume_camera_pose
        self._plotter.camera_position = volume_camera_pose(self._vol.shape, self._affine)
        self._plotter.camera.view_angle = 30.
        self._plotter.reset_camera()
        self._plotter.camera.zoom(1.2)
        self._camera = self._plotter.camera_position
        self._plotter.render()

    def move_to(self, x, y, z):
        """从外部 (如 2D 拖动) 更新切面位置并刷新"""
        if self._vol is None:
            return
        x = int(np.clip(x, 0, self._vol.shape[0] - 1))
        y = int(np.clip(y, 0, self._vol.shape[1] - 1))
        z = int(np.clip(z, 0, self._vol.shape[2] - 1))
        self._slice = {"x": x, "y": y, "z": z}
        self._refresh_slices_only()

    # ── 完整重建 ──
    def _full_rebuild(self):
        self._rebuilding=True
        vol = self._vol
        x, y, z = self._slice["x"], self._slice["y"], self._slice["z"]
        try:
            if self._plotter and not self._first_render:
                self._camera = self._plotter.camera_position
        except Exception:
            self._camera = None

        self._plotter.clear_plane_widgets()
        self._plotter.clear()
        sh = vol.shape
        self._step = 2 if max(sh) > 160 else 1
        # 添加三个可拖动 plane widget 并保留引用
        self._pw = []
        from ui.volume_geometry import volume_world_bounds,slice_plane_pose
        bounds=volume_world_bounds(sh,self._affine)
        for axis, index, color in [("x",x,"red"),("y",y,"lime"),("z",z,"dodgerblue")]:
            origin,normal=slice_plane_pose(sh,self._affine,("x","y","z").index(axis),index)
            pw = self._plotter.add_plane_widget(
                callback=lambda n, o, a=axis: self._on_drag(a, n, o),
                normal=normal, origin=origin, bounds=bounds,
                color=color, outline_translation=False, normal_rotation=False,
            )
            # 半透明：用户看到也抓到平面本身
            pw.GetPlaneProperty().SetOpacity(0.05)
            pw.GetSelectedPlaneProperty().SetOpacity(0.12)
            pw.GetPlaneProperty().SetEdgeVisibility(0)
            pw.GetOutlineProperty().SetColor(.45,.53,.65)
            pw.GetOutlineProperty().SetOpacity(.12)
            self._pw.append((pw, axis))

        self._refresh_slices_only()

        try:
            if self._camera:
                self._plotter.camera_position = self._camera
            elif self._first_render:
                self.reset_view()
                self._first_render = False
        except Exception:
            pass
        self._need_full_rebuild = False
        self._rebuilding=False

    # ── 仅重建切片网格（不碰 plane widget） ──
    def _refresh_slices_only(self):
        from ui.plane_colors import slice_rgba
        from ui.volume_geometry import slice_plane_coordinates,slice_plane_pose
        indices=[self._slice[key] for key in ('x','y','z')]
        shape=self._vol.shape
        clim=(float(np.nanmin(self._vol)),float(np.nanmax(self._vol))) if self._vmin is None else (self._vmin,self._vmax)
        for axis,name in enumerate(('sagittal','coronal','axial')):
            # Preserve the exact selected plane; downsample only within it.
            selected=[slice(None,None,self._step)]*3;selected[axis]=indices[axis]
            selected=tuple(selected)
            base=self._vol[selected]
            overlays=[{**ov,'data':ov['volume'][selected]} for ov in self._ov or []]
            rgba=slice_rgba(base,overlays,*clim)
            coordinates=slice_plane_coordinates(shape,self._affine,axis,indices[axis],self._step)
            plane=pv.StructuredGrid(*coordinates)
            plane.point_data['slice_rgba']=rgba.reshape((-1,4),order='F')
            actor=self._plotter.add_mesh(plane,scalars='slice_rgba',rgb=True,
                lighting=False,name='plane_'+name,show_scalar_bar=False,reset_camera=False)
            actor._name='plane_'+name
        for widget,axis_name in self._pw:
            axis=("x","y","z").index(axis_name)
            origin,normal=slice_plane_pose(shape,self._affine,axis,indices[axis])
            widget.SetOrigin(origin); widget.SetNormal(normal)
        self._plotter.render()

    # ── 拖动回调 ──
    def _on_drag(self, axis, normal, origin):
        if self._rebuilding or self._vol is None: return
        idx_map = {"x": 0, "y": 1, "z": 2}
        idx = idx_map[axis]
        voxel_origin=self._inverse_affine[:3,:3]@np.asarray(origin)+self._inverse_affine[:3,3]
        val = int(np.clip(round(voxel_origin[idx]), 0, self._vol.shape[idx] - 1))
        self._slice[axis] = val
        self._refresh_slices_only()
        self.slice_changed.emit(
            self._slice["x"], self._slice["y"], self._slice["z"])
