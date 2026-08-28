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
        self.fig.subplots_adjust(left=0.02, right=0.90, top=0.90, bottom=0.02)
        self.mpl_connect("button_press_event", self.on_mouse_press)
        self.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.mpl_connect("button_release_event", self.on_mouse_release)

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
        self.ax.set_title(self.title, color="white", fontsize=12,
                          fontweight="bold", pad=10)
        if data_slice is None:
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
                    self.add_parallel_colorbar(im, i)
        if crosshair:
            cx, cy = crosshair
            self.ax.axhline(cy, color="lime", linewidth=0.8,
                            linestyle="--", alpha=0.6)
            self.ax.axvline(cx, color="lime", linewidth=0.8,
                            linestyle="--", alpha=0.6)
            self.ax.scatter([cx], [cy], s=15, c="yellow",
                            edgecolors="black", linewidths=0.5, zorder=10)
        self.draw_idle()

    def add_parallel_colorbar(self, mappable, index):
        ax_ins = inset_axes(
            self.ax, width="5%", height="60%", loc="lower left",
            bbox_to_anchor=(1.05 + index * 0.12, 0.2, 0.1, 0.6),
            bbox_transform=self.ax.transAxes, borderpad=0)
        cb = self.fig.colorbar(mappable, cax=ax_ins)
        cb.ax.yaxis.set_tick_params(color="white", labelcolor="white",
                                    labelsize=7)
        cb.outline.set_edgecolor("white")
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
        self._step = 2

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

    def render_3d(self, volume, x, y, z, overlays=None, vmin=None, vmax=None):
        if not self.available or volume is None:
            return
        # Packaged VTK builds can crash natively when repeatedly slicing
        # overlay grids. Keep 3D in stable anatomical-slice mode.
        overlays = None
        x = int(np.clip(x, 0, volume.shape[0] - 1))
        y = int(np.clip(y, 0, volume.shape[1] - 1))
        z = int(np.clip(z, 0, volume.shape[2] - 1))
        new_data = (
            volume is not self._vol
            or overlays is not self._ov
            or vmin != self._vmin
            or vmax != self._vmax
        )
        self._vol, self._ov = volume, overlays
        self._vmin, self._vmax = vmin, vmax
        self._slice = {"x": x, "y": y, "z": z}

        if new_data or getattr(self, "_need_full_rebuild", True):
            self._full_rebuild()
        else:
            self._refresh_slices_only()

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
        vol = self._vol
        x, y, z = self._slice["x"], self._slice["y"], self._slice["z"]
        try:
            if self._plotter and not self._first_render:
                self._camera = self._plotter.camera_position
        except Exception:
            self._camera = None

        self._plotter.clear()
        sh = vol.shape
        self._step = 2 if max(sh) > 160 else 1
        vs = vol[::self._step, ::self._step, ::self._step]
        self._grid = pv.ImageData(
            dimensions=vs.shape, spacing=(self._step,) * 3)
        self._grid.point_data["base"] = vs.flatten(order="F")
        clim = (float(np.nanmin(vol)), float(np.nanmax(vol))) if self._vmin is None else (float(self._vmin), float(self._vmax))

        # 添加三个可拖动 plane widget 并保留引用
        self._pw = []
        for axis, origin, color, label in [
            ("x", (x, 0, 0), "red", "矢状面"),
            ("y", (0, y, 0), "lime", "冠状面"),
            ("z", (0, 0, z), "dodgerblue", "轴向面"),
        ]:
            pw = self._plotter.add_plane_widget(
                callback=lambda n, o, a=axis: self._on_drag(a, n, o),
                normal=axis, origin=origin,
                bounds=(0, sh[0] - 1, 0, sh[1] - 1, 0, sh[2] - 1),
                color=color, outline_translation=False,
            )
            # 半透明：用户看到也抓到平面本身
            pw.GetPlaneProperty().SetOpacity(0.05)
            pw.GetSelectedPlaneProperty().SetOpacity(0.12)
            pw.GetPlaneProperty().SetEdgeVisibility(0)
            self._pw.append((pw, axis))

        # 叠加层网格缓存
        if self._ov:
            self._ov_grids = []
            for ov in self._ov:
                od = ov["volume"][::self._step, ::self._step, ::self._step]
                og = pv.ImageData(dimensions=od.shape, spacing=(self._step,) * 3)
                og.point_data["efield"] = od.flatten(order="F")
                self._ov_grids.append((og, ov))
        else:
            self._ov_grids = []

        self._refresh_slices_only()

        try:
            if self._camera:
                self._plotter.camera_position = self._camera
            elif self._first_render:
                self._plotter.view_isometric()
                self._plotter.camera.zoom(1.2)
                self._first_render = False
        except Exception:
            pass
        self._need_full_rebuild = False

    # ── 仅重建切片网格（不碰 plane widget） ──
    def _refresh_slices_only(self):
        x, y, z = self._slice["x"], self._slice["y"], self._slice["z"]

        # 移除旧的切片 actor
        to_remove = []
        for a in list(self._plotter.renderer.actors.values()):
            if hasattr(a, '_name') and (
                a._name.endswith(("_base", "_edge", "_ov")) or
                a._name.startswith("ov_")
            ):
                to_remove.append(a)
        for a in to_remove:
            self._plotter.remove_actor(a, reset_camera=False)

        clim = (float(np.nanmin(self._vol)), float(np.nanmax(self._vol))
                ) if self._vmin is None else (float(self._vmin), float(self._vmax))

        for name, axis, origin, color in [
            ("slc_sagittal", "x", (x, 0, 0), "lime"),
            ("slc_coronal",  "y", (0, y, 0), "lime"),
            ("slc_axial",    "z", (0, 0, z), "lime"),
        ]:
            slc = self._grid.slice(normal=axis, origin=origin)
            actor = self._plotter.add_mesh(
                slc, cmap="gray", clim=clim, lighting=False,
                name=name + "_base", show_scalar_bar=False)
            actor._name = name + "_base"
            edges = slc.extract_feature_edges(
                boundary_edges=True, feature_edges=False,
                manifold_edges=False)
            if edges.n_points > 0:
                eactor = self._plotter.add_mesh(
                    edges, color=color, line_width=3,
                    name=name + "_edge")
                eactor._name = name + "_edge"

        # 叠加层
        for og, ov in self._ov_grids:
            is_atlas = ov.get("is_atlas", False)
            for idx, (name, axis, origin, color) in enumerate([
                ("ov_sag", "x", (x, 0, 0), None),
                ("ov_cor", "y", (0, y, 0), None),
                ("ov_axi", "z", (0, 0, z), None),
            ]):
                try:
                    osl = og.slice(normal=axis, origin=origin)
                    if is_atlas:
                        vv = osl.point_data["efield"]
                        osl.point_data["efield"][vv == 0] = np.nan
                    else:
                        osl = osl.threshold(ov["min"] + 1e-6, scalars="efield")
                    if osl.n_points > 0:
                        actor_name = f"{name}_{id(ov)}_ov"
                        oactor = self._plotter.add_mesh(
                            osl, scalars="efield", cmap=ov["cmap"],
                            opacity=ov.get("alpha", 0.7),
                            lighting=False, name=actor_name,
                            show_scalar_bar=False,
                            clim=(ov["min"], ov["max"])
                            if not is_atlas else None)
                        oactor._name = actor_name
                except Exception:
                    pass

        self._plotter.render()

    # ── 拖动回调 ──
    def _on_drag(self, axis, normal, origin):
        idx_map = {"x": 0, "y": 1, "z": 2}
        idx = idx_map[axis]
        val = int(np.clip(round(origin[idx]), 0, self._vol.shape[idx] - 1))
        self._slice[axis] = val
        self._refresh_slices_only()
        self.slice_changed.emit(
            self._slice["x"], self._slice["y"], self._slice["z"])
