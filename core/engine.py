import os
import traceback
import nibabel as nib
import numpy as np

try:
    from nibabel.processing import resample_from_to
except Exception:
    resample_from_to = None

try:
    from scipy.ndimage import zoom
except Exception:
    zoom = None


class NiiEngine:
    def __init__(self):
        self.image = None
        self.data = None
        self.affine = None
        self.overlays = []
        self.display_min = None
        self.display_max = None

    def _to_3d_image(self, img):
        img = nib.as_closest_canonical(img)
        data = img.get_fdata()
        if data.ndim == 4:
            data = data[:, :, :, 0]
            img = nib.Nifti1Image(data, img.affine, header=img.header)
        if data.ndim != 3:
            raise ValueError(f"Only 3D/4D supported, got {data.ndim}D")
        data = np.asarray(data, dtype=np.float32)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        return img, data

    def _calc_display_range(self, data):
        finite_data = data[np.isfinite(data)]
        if finite_data.size > 0:
            display_min = float(np.nanpercentile(finite_data, 1))
            display_max = float(np.nanpercentile(finite_data, 99))
            if display_max <= display_min:
                display_min = float(np.nanmin(finite_data))
                display_max = float(np.nanmax(finite_data))
            if display_max <= display_min:
                display_max = display_min + 1e-6
        else:
            display_min, display_max = 0.0, 1.0
        return display_min, display_max

    def _same_space(self, ov_data, ov_affine):
        if self.data is None or self.affine is None:
            return False
        same_shape = ov_data.shape == self.data.shape
        same_affine = np.allclose(ov_affine, self.affine, atol=1e-4)
        return same_shape and same_affine

    def file_matches_base_space(self, file_path):
        if self.data is None or self.affine is None:
            return False
        try:
            img = nib.as_closest_canonical(nib.load(file_path))
            shape = img.shape[:3]
            return shape == self.data.shape and np.allclose(img.affine, self.affine, atol=1e-4)
        except Exception:
            return False

    @staticmethod
    def files_match_space(file_a, file_b):
        try:
            img_a = nib.as_closest_canonical(nib.load(file_a))
            img_b = nib.as_closest_canonical(nib.load(file_b))
            return (
                img_a.shape[:3] == img_b.shape[:3] and
                np.allclose(img_a.affine, img_b.affine, atol=1e-4)
            )
        except Exception:
            return False

    def _is_label_map(self, data):
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            return False
        nonzero = finite[finite != 0]
        if nonzero.size == 0:
            return False
        sample_source = nonzero
        n_sample = min(10000, sample_source.size)
        sample = sample_source[np.linspace(0, sample_source.size - 1, n_sample, dtype=int)]
        sample_unique = np.unique(sample)
        return (
            len(sample_unique) < 1000 and
            np.all(np.isfinite(sample_unique)) and
            np.allclose(sample_unique, np.round(sample_unique), atol=1e-4)
        )

    def _resample_overlay_to_base(self, ov_img, ov_data, is_label_map=False):
        if self._same_space(ov_data, ov_img.affine):
            return ov_data.astype(np.float32)

        if resample_from_to is not None:
            try:
                target = (self.data.shape, self.affine)
                order = 0 if is_label_map else 1
                resampled_img = resample_from_to(ov_img, target, order=order)
                resampled_data = resampled_img.get_fdata().astype(np.float32)
                resampled_data = np.nan_to_num(resampled_data, nan=0.0, posinf=0.0, neginf=0.0)
                if is_label_map:
                    resampled_data = np.rint(resampled_data).astype(np.float32)
                if resampled_data.shape == self.data.shape:
                    return resampled_data
            except Exception:
                pass

        raise RuntimeError(
            "Overlay cannot be resampled to the base image. "
            "Please use an overlay in the same anatomical space, or install scipy for physical-space resampling."
        )

    # ----------------------------------------------------------------------
    def load_label_dict(self, overlay_idx, file_path):
        if overlay_idx < 0 or overlay_idx >= len(self.overlays):
            return False
        try:
            label_dict = {}
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    clean_line = line.strip().replace(',', ' ').replace('\t', ' ')
                    if not clean_line or clean_line.startswith('#'):
                        continue
                    parts = clean_line.split()
                    if len(parts) >= 2 and parts[0].isdigit():
                        label_dict[int(parts[0])] = " ".join(parts[1:])
            self.overlays[overlay_idx]["label_dict"] = label_dict
            return True
        except Exception:
            return False

    def load_image(self, file_path):
        try:
            raw_img = nib.load(file_path)
            img, data = self._to_3d_image(raw_img)
            self.image = img
            self.data = data
            self.affine = img.affine
            self.overlays = []
            self.display_min, self.display_max = self._calc_display_range(self.data)
            return True, self.data.shape
        except Exception as e:
            return False, f"{e}\n{traceback.format_exc()}"

    def load_overlay(self, file_path):
        try:
            if self.image is None or self.data is None:
                return False, "Load base image first."

            raw_ov_img = nib.load(file_path)
            ov_img, ov_data = self._to_3d_image(raw_ov_img)
            source_is_label_map = self._is_label_map(ov_data)
            ov_data = self._resample_overlay_to_base(ov_img, ov_data, is_label_map=source_is_label_map)
            ov_data = np.nan_to_num(ov_data, nan=0.0, posinf=0.0, neginf=0.0)

            finite_data = ov_data[np.isfinite(ov_data)]
            if finite_data.size > 0:
                ov_min = float(np.nanmin(finite_data))
                ov_max = float(np.nanmax(finite_data))
            else:
                ov_min, ov_max = 0.0, 1.0
            if ov_max <= ov_min:
                ov_max = ov_min + 1e-6

            is_atlas = self._is_label_map(ov_data) or source_is_label_map
            unique_vals = np.unique(ov_data) if is_atlas else np.array([])

            label_dict = {}
            if is_atlas:
                bp = file_path
                if bp.endswith('.nii.gz'):
                    bp = bp[:-7]
                elif bp.endswith('.nii'):
                    bp = bp[:-4]
                tp = bp + ".txt"
                if os.path.exists(tp):
                    with open(tp, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip().replace(',', ' ').replace('\t', ' ')
                            if not line or line.startswith('#'):
                                continue
                            parts = line.split()
                            if len(parts) >= 2 and parts[0].isdigit():
                                label_dict[int(parts[0])] = " ".join(parts[1:])

            cmaps = ["hot", "cool", "jet", "viridis", "magma", "winter", "plasma", "inferno", "turbo"]
            idx = len(self.overlays)
            overlay_config = {
                "name": os.path.basename(file_path),
                "data": ov_data.astype(np.float32),
                "is_atlas": is_atlas,
                "labels": unique_vals.astype(int).tolist() if is_atlas else [],
                "active_labels": unique_vals.astype(int).tolist() if is_atlas else [],
                "label_dict": label_dict,
                "min": ov_min,
                "max": ov_max,
                "cmap": "tab20" if is_atlas else cmaps[idx % len(cmaps)],
                "alpha": 0.4 if is_atlas else 0.7,
            }
            self.overlays.append(overlay_config)
            return True, len(self.overlays) - 1

        except Exception as e:
            return False, f"{e}\n{traceback.format_exc()}"

    def voxel_to_mni(self, x, y, z):
        if self.affine is None:
            return 0.0, 0.0, 0.0
        mni = self.affine @ np.array([x, y, z, 1])
        return tuple(mni[:3])

    def mni_to_voxel(self, mni_x, mni_y, mni_z):
        """将 MNI152 空间坐标反变换为体素坐标"""
        if self.affine is None:
            sh = self.data.shape
            return sh[0] // 2, sh[1] // 2, sh[2] // 2
        inv_affine = np.linalg.inv(self.affine)
        voxel = inv_affine @ np.array([mni_x, mni_y, mni_z, 1])
        return int(round(voxel[0])), int(round(voxel[1])), int(round(voxel[2]))
