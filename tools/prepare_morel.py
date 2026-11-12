"""Prepare the 1 mm Morel atlas and preserve its overlapping source masks."""
from argparse import ArgumentParser
from collections import defaultdict
import gzip
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
from scipy.ndimage import binary_fill_holes


def prepare(archive_path, mask_path, output):
    labels = {}
    memberships = defaultdict(list)
    with ZipFile(archive_path) as archive:
        template_bytes = archive.read('MNI152_T1_1mm.nii.gz')
        template = nib.Nifti1Image.from_bytes(gzip.decompress(template_bytes))
        notice = archive.read('Readme.txt')
        for hemisphere, side in (('left', 'L'), ('right', 'R')):
            prefix = f'{hemisphere}-vols-1mm/'
            for member in sorted(archive.namelist()):
                if not (member.startswith(prefix) and member.endswith('.nii.gz')):
                    continue
                name = PurePosixPath(member).name.removesuffix('.nii.gz')
                if name in {'global', 'MAX_VOLUME'}:
                    continue
                image = nib.Nifti1Image.from_bytes(gzip.decompress(archive.read(member)))
                data = np.asarray(image.dataobj)
                if image.shape != template.shape or not np.allclose(image.affine, template.affine):
                    raise ValueError(f'Template grid mismatch in {member}')
                if not np.isin(data, [0, 1]).all() or not data.any():
                    raise ValueError(f'Expected a nonempty binary mask in {member}')
                region = len(labels) + 1
                labels[region] = f'{"STN" if name == "STh" else name}_{side}'
                for voxel in np.flatnonzero(data):
                    memberships[voxel].append(region)

    if not {'STN_L', 'STN_R'} <= set(labels.values()):
        raise ValueError('The archive must contain both STN masks')
    atlas = np.zeros(template.shape, dtype=np.int16)
    code_by_membership = {(region,): region for region in labels}
    for voxel, regions in sorted(memberships.items()):
        code = code_by_membership.setdefault(tuple(regions), len(code_by_membership) + 1)
        atlas.flat[voxel] = code
    voxel_labels = {region: [] for region in labels}
    for regions, code in code_by_membership.items():
        for region in regions:
            voxel_labels[region].append(code)

    mask = binary_fill_holes(np.asarray(resample_from_to(nib.load(mask_path), template, order=0).dataobj) > 0)
    if not mask.any() or np.count_nonzero(mask & (atlas > 0)) < .98 * len(memberships):
        raise ValueError('Brain mask is incompatible with the Morel template')

    output.mkdir(parents=True, exist_ok=True)
    (output / 'Morel_T1_1mm.nii.gz').write_bytes(template_bytes)
    (output / 'Morel_upstream_readme.txt').write_bytes(notice)
    nib.save(nib.Nifti1Image(atlas, template.affine), output / 'Morel_1mm.nii.gz')
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), template.affine), output / 'Morel_template_mask.nii.gz')
    for name, table in (('labels', labels), ('voxel_labels', voxel_labels)):
        (output / f'Morel_{name}.json').write_text(json.dumps(table, indent=2) + '\n', encoding='utf-8')
    print(f'{len(labels)} Morel regions and {len(code_by_membership)} voxel label codes')


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path, help='MorelAtlasMNI152.zip from Zenodo')
    parser.add_argument('mask', type=Path, help='MNI152NLin6Asym res-01 brain mask from TemplateFlow')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'assets/atlases')
    args = parser.parse_args()
    prepare(args.archive, args.mask, args.output)
