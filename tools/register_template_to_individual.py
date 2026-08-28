import argparse
import shutil
import tempfile
import uuid
from pathlib import Path

import SimpleITK as sitk


def _path_suffix(path):
    name = Path(path).name
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    return Path(path).suffix


def _is_ascii_path(path):
    try:
        str(path).encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def _sitk_temp_dir():
    path = Path(tempfile.gettempdir()) / "neuroguide_sitk_paths"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _stage_input_for_sitk(path):
    path = Path(path)
    if _is_ascii_path(path):
        return path
    staged = _sitk_temp_dir() / f"in_{uuid.uuid4().hex}{_path_suffix(path)}"
    shutil.copy2(path, staged)
    return staged


def _write_image_for_sitk(image, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if _is_ascii_path(output_path):
        sitk.WriteImage(image, str(output_path))
        return
    staged = _sitk_temp_dir() / f"out_{uuid.uuid4().hex}{_path_suffix(output_path)}"
    sitk.WriteImage(image, str(staged))
    shutil.copy2(staged, output_path)


def _write_transform_for_sitk(transform, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if _is_ascii_path(output_path):
        sitk.WriteTransform(transform, str(output_path))
        return
    staged = _sitk_temp_dir() / f"transform_{uuid.uuid4().hex}.tfm"
    sitk.WriteTransform(transform, str(staged))
    shutil.copy2(staged, output_path)


def read_image(path):
    return sitk.ReadImage(str(_stage_input_for_sitk(path)))


def read_float_image(path):
    image = read_image(path)
    return sitk.Cast(image, sitk.sitkFloat32)


def normalize_for_registration(image):
    image = sitk.Clamp(image, lowerBound=0.0)
    return sitk.Normalize(image)


def shrink_for_nonlinear(image, factors=(4, 4, 4)):
    return sitk.Shrink(image, factors)


def register_stage(fixed, moving, initial_transform, transform_name, moving_initial_transform=None):
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.02, seed=20260707)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=1e-3,
        numberOfIterations=100,
        relaxationFactor=0.5,
        gradientMagnitudeTolerance=1e-6,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel([4, 2, 1])
    registration.SetSmoothingSigmasPerLevel([2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    if moving_initial_transform is not None:
        registration.SetMovingInitialTransform(moving_initial_transform)
    registration.SetInitialTransform(initial_transform, inPlace=False)
    print(f"Running {transform_name} registration...")
    final_transform = registration.Execute(fixed, moving)
    print(
        f"{transform_name}: metric={registration.GetMetricValue():.6f}, "
        f"stop='{registration.GetOptimizerStopConditionDescription()}'"
    )
    return final_transform


def image_physical_center(image):
    size = image.GetSize()
    continuous_index = [(s - 1) / 2.0 for s in size]
    return image.TransformContinuousIndexToPhysicalPoint(continuous_index)


def make_affine_delta(fixed):
    transform = sitk.AffineTransform(3)
    transform.SetCenter(image_physical_center(fixed))
    return transform


def register_bspline_stage(fixed, moving, moving_initial_transform):
    fixed_low = shrink_for_nonlinear(fixed)
    moving_low = shrink_for_nonlinear(moving)
    mesh_size = [5, 5, 5]
    bspline_init = sitk.BSplineTransformInitializer(fixed_low, mesh_size, order=3)

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.02, seed=20260707)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5,
        numberOfIterations=25,
        maximumNumberOfCorrections=5,
        maximumNumberOfFunctionEvaluations=120,
        costFunctionConvergenceFactor=1e7,
    )
    registration.SetShrinkFactorsPerLevel([2, 1])
    registration.SetSmoothingSigmasPerLevel([1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetMovingInitialTransform(moving_initial_transform)
    registration.SetInitialTransform(bspline_init, inPlace=False)

    print("Running fast bspline registration...")
    bspline = registration.Execute(fixed_low, moving_low)
    print(
        f"bspline: metric={registration.GetMetricValue():.6f}, "
        f"stop='{registration.GetOptimizerStopConditionDescription()}'"
    )
    return bspline


def register_template_to_individual(fixed_t1, moving_template_t1, use_nonlinear=True):
    fixed = normalize_for_registration(read_float_image(fixed_t1))
    moving = normalize_for_registration(read_float_image(moving_template_t1))

    try:
        rigid_init = sitk.CenteredTransformInitializer(
            fixed,
            moving,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
        rigid = register_stage(fixed, moving, rigid_init, "rigid")
    except Exception as exc:
        print(f"rigid geometry initialization failed: {exc}")
        rigid_init = sitk.CenteredTransformInitializer(
            fixed,
            moving,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.MOMENTS,
        )
        rigid = register_stage(fixed, moving, rigid_init, "rigid_moments")

    composite = sitk.CompositeTransform(3)
    add_transform_flattened(composite, rigid)

    try:
        affine_delta = register_stage(
            fixed,
            moving,
            make_affine_delta(fixed),
            "affine",
            moving_initial_transform=rigid,
        )
        add_transform_flattened(composite, affine_delta)
    except Exception as exc:
        print(f"affine registration failed; using rigid only: {exc}")

    if use_nonlinear:
        try:
            bspline = register_bspline_stage(fixed, moving, composite)
            add_transform_flattened(composite, bspline)
        except Exception as exc:
            print(f"bspline registration failed; using previous transform: {exc}")
    return composite


def add_transform_flattened(composite, transform):
    if isinstance(transform, sitk.CompositeTransform):
        for index in range(transform.GetNumberOfTransforms()):
            add_transform_flattened(composite, transform.GetNthTransform(index))
    else:
        composite.AddTransform(transform)


def resample_to_fixed(moving_path, fixed_reference_path, transform, output_path, is_label=False):
    fixed = read_image(fixed_reference_path)
    moving = read_image(moving_path)
    interpolator = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    default_value = 0
    resampled = sitk.Resample(
        moving,
        fixed,
        transform,
        interpolator,
        default_value,
        moving.GetPixelID(),
    )
    _write_image_for_sitk(resampled, output_path)


def write_transform(transform, output_path):
    _write_transform_for_sitk(transform, output_path)


def looks_like_label(path):
    name = path.name.lower()
    return "aal" in name or "label" in name or "atlas" in name or "mask" in name


def main():
    parser = argparse.ArgumentParser(
        description="Register NeuroGuide template-space overlays to an individual T1."
    )
    parser.add_argument("--fixed-t1", required=True, help="Individual/native T1 NIfTI.")
    parser.add_argument(
        "--template-t1",
        default=str(Path(__file__).resolve().parents[1] / "mri" / "T1.nii"),
        help="Template T1 that the overlays are aligned to.",
    )
    parser.add_argument(
        "--overlay",
        action="append",
        required=True,
        help="Overlay NIfTI aligned with --template-t1. Repeat for multiple overlays.",
    )
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="Overlay filename/path that must use nearest-neighbor interpolation.",
    )
    args = parser.parse_args()

    fixed_t1 = Path(args.fixed_t1)
    template_t1 = Path(args.template_t1)
    out_dir = Path(args.out_dir)
    label_names = {Path(p).name for p in args.label}

    transform = register_template_to_individual(fixed_t1, template_t1)
    transform_path = out_dir / "template_to_individual_transform.tfm"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_transform(transform, transform_path)
    print(f"Saved transform: {transform_path}")

    for overlay_arg in args.overlay:
        overlay = Path(overlay_arg)
        is_label = overlay.name in label_names or looks_like_label(overlay)
        suffix = ".nii.gz" if overlay.name.endswith(".nii.gz") else overlay.suffix
        stem = overlay.name[:-7] if overlay.name.endswith(".nii.gz") else overlay.stem
        output = out_dir / f"{stem}_in_individual_space{suffix}"
        resample_to_fixed(overlay, fixed_t1, transform, output, is_label=is_label)
        mode = "nearest-neighbor" if is_label else "linear"
        print(f"Saved {output} ({mode})")


if __name__ == "__main__":
    main()
