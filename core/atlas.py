"""Atlas labels and independently cached template-to-native ANTs registration."""
from __future__ import annotations
import hashlib
import json
import os
import time
from pathlib import Path
import nibabel as nib
import numpy as np

ATLAS_NAMES = {'aal3': 'AAL3', 'morel': 'Morel'}

TRANSLATIONS = {
    'Precentral':'中央前回','Postcentral':'中央后回','Hippocampus':'海马','ParaHippocampal':'海马旁回',
    'Amygdala':'杏仁核','Insula':'岛叶','Caudate':'尾状核','Putamen':'壳核','Pallidum':'苍白球',
    'Frontal_Sup':'额上回','Frontal_Mid':'额中回','Frontal_Inf_Oper':'额下回盖部','Frontal_Inf_Tri':'额下回三角部',
    'Frontal_Inf_Orb':'额下回眶部','Frontal_Sup_Medial':'内侧额上回','Frontal_Med_Orb':'内侧眶额回',
    'Frontal_Sup_Orb':'眶额上回','Frontal_Mid_Orb':'眶额中回','Rolandic_Oper':'中央沟盖',
    'Supp_Motor_Area':'辅助运动区','Olfactory':'嗅皮层','Rectus':'直回','Cingulate_Mid':'中扣带回',
    'Cingulate_Post':'后扣带回','Cingulate_Ant':'前扣带回','Calcarine':'距状裂皮层','Cuneus':'楔叶',
    'Lingual':'舌回','Occipital_Sup':'枕上回','Occipital_Mid':'枕中回','Occipital_Inf':'枕下回',
    'Fusiform':'梭状回','Parietal_Sup':'顶上小叶','Parietal_Inf':'顶下小叶','SupraMarginal':'缘上回',
    'Angular':'角回','Precuneus':'楔前叶','Paracentral_Lobule':'中央旁小叶','Heschl':'颞横回',
    'Temporal_Sup':'颞上回','Temporal_Pole_Sup':'颞极上部','Temporal_Mid':'颞中回',
    'Temporal_Pole_Mid':'颞极中部','Temporal_Inf':'颞下回','N_Acc':'伏隔核','SN_pc':'黑质致密部',
    'SN_pr':'黑质网状部','VTA':'腹侧被盖区','Red_N':'红核','LC':'蓝斑','Raphe_D':'中缝背核','Raphe_M':'中缝正中核',
    'ACC_sub':'膝下前扣带','ACC_pre':'膝前前扣带','ACC_sup':'膝上前扣带',
    'OFCmed':'内侧眶额皮层','OFCant':'前眶额皮层','OFCpost':'后眶额皮层','OFClat':'外侧眶额皮层',
    # Morel v1.0 abbreviations, with anatomical subdivisions kept distinct.
    'AD': '丘脑前背侧核',
    'AM': '丘脑前内侧核',
    'AV': '丘脑前腹侧核',
    'CL': '丘脑中央外侧核',
    'CM': '丘脑中央中核',
    'CeM': '丘脑中央内侧核',
    'Hb': '缰核',
    'LD': '丘脑外侧背核',
    'LGNmc': '外侧膝状体大细胞层',
    'LGNpc': '外侧膝状体小细胞层',
    'LP': '丘脑外侧后核',
    'Li': '丘脑界限核',
    'MDmc': '丘脑内侧背核大细胞部',
    'MDpc': '丘脑内侧背核小细胞部',
    'MGN': '内侧膝状体核',
    'MV': '丘脑内侧腹核',
    'Pf': '丘脑束旁核',
    'Po': '丘脑后核',
    'PuA': '丘脑枕前部',
    'PuI': '丘脑枕下部',
    'PuL': '丘脑枕外侧部',
    'PuM': '丘脑枕内侧部',
    'Pv': '丘脑室旁核',
    'RN': '红核',
    'SG': '丘脑膝上核',
    'STN': '丘脑底核',
    'VAmc': '丘脑腹前核大细胞部',
    'VApc': '丘脑腹前核小细胞部',
    'VLa': '丘脑腹外侧前核',
    'VLpd': '丘脑腹外侧后核背侧部',
    'VLpv': '丘脑腹外侧后核腹侧部',
    'VM': '丘脑腹内侧核',
    'VPI': '丘脑腹后下核',
    'VPLa': '丘脑腹后外侧核前部',
    'VPLp': '丘脑腹后外侧核后部',
    'VPM': '丘脑腹后内侧核',
    'mtt': '乳头丘脑束',
    'sPf': '丘脑束旁下核',
    'thalamus_body': '丘脑主体',
}

def region_name(name):
    side = '左' if name.endswith('_L') else '右' if name.endswith('_R') else ''
    base = name[:-2] if side else name
    translated = TRANSLATIONS.get(base)
    if translated is None and base.startswith('Cerebelum_'): translated = '小脑 ' + base[10:].replace('_','/')
    if translated is None and base.startswith('Vermis_'): translated = '小脑蚓部 ' + base[7:].replace('_','/')
    if translated is None and base.startswith('Thal_'): translated = '丘脑 ' + base[5:]
    return side + translated if translated else name

def load_labels(resources, atlas_id='aal3'):
    path = Path(resources) / (ATLAS_NAMES[atlas_id] + '_labels.json')
    return {int(k): v for k, v in json.loads(path.read_text(encoding='utf-8')).items()}


def atlas_voxel_labels(record, region_ids):
    memberships = record.get('voxel_labels', {})
    return sorted({value for region in region_ids for value in memberships.get(str(region), [region])})


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''): digest.update(block)
    return digest.hexdigest()

def validate_labels(image, reference, valid_labels):
    if image.shape != reference.shape or not np.allclose(image.affine,reference.affine,atol=1e-4):
        raise ValueError('图谱与个体 MRI 网格不一致。')
    data = image.get_fdata(dtype=np.float32)
    if not np.all(np.isfinite(data)) or not np.allclose(data,np.rint(data),atol=1e-5):
        raise ValueError('映射图谱包含无效或非整数标签。')
    found = set(np.unique(data).astype(int)) - {0}
    if not found or not found.issubset(set(valid_labels)):
        raise ValueError('映射图谱的脑区编号不正确。')
    return np.rint(data).astype(np.int16)



def nifti_to_ants(image):
    """Use the full NIfTI affine, including legacy shear, without ITK file IO.

    NIfTI physical coordinates are RAS; ANTs physical coordinates are LPS.
    Array axes remain x/y/z. No reorientation or MRI interpolation occurs.
    """
    import ants
    affine=np.asarray(image.affine,dtype=float)
    spacing=np.linalg.norm(affine[:3,:3],axis=0)
    if (len(image.shape)!=3 or not np.isfinite(affine).all() or np.any(spacing<=0)
            or abs(np.linalg.det(affine[:3,:3]))<1e-8):
        raise ValueError("Invalid native NIfTI grid for atlas registration")
    ras_to_lps=np.diag([-1.,-1.,1.])
    data=image.get_fdata(dtype=np.float32)
    return ants.from_numpy(data,origin=tuple(ras_to_lps@affine[:3,3]),
                           spacing=tuple(spacing),direction=ras_to_lps@(affine[:3,:3]/spacing))


def ants_on_reference(image,reference):
    """Export an ANTs result on its exact fixed grid with the original affine."""
    affine=np.asarray(reference.affine,dtype=float)
    spacing=np.linalg.norm(affine[:3,:3],axis=0)
    ras_to_lps=np.diag([-1.,-1.,1.])
    if (tuple(image.shape)!=reference.shape
            or not np.allclose(image.spacing,spacing,atol=1e-5,rtol=0)
            or not np.allclose(image.origin,ras_to_lps@affine[:3,3],atol=1e-5,rtol=0)
            or not np.allclose(image.direction,ras_to_lps@(affine[:3,:3]/spacing),atol=1e-5,rtol=0)):
        raise ValueError("Atlas result does not match the fixed physical grid")
    header=reference.header.copy();header.set_data_dtype(np.float32);header.set_slope_inter(1,0)
    result=nib.Nifti1Image(image.numpy().astype(np.float32),affine,header)
    result.set_sform(affine,code=max(1,int(header["sform_code"])));result.set_qform(None,code=0)
    return result




def atlas_working_grid(reference,brain):
    """World-aligned 2 mm grid for registration only; native MRI stays unchanged."""
    from nibabel.processing import resample_to_output,resample_from_to
    fixed=resample_to_output(reference,voxel_sizes=(2.,2.,2.),order=1)
    source_mask=nib.Nifti1Image(np.asarray(brain,dtype=np.uint8),reference.affine)
    mask=resample_from_to(source_mask,fixed,order=0)
    return nifti_to_ants(fixed),nifti_to_ants(mask)




def preprocessing_affine(task,reference_path):
    """Find the audited physical subject-to-MNI affine from electrode placement."""
    task=Path(task).resolve();reference_path=Path(reference_path).resolve()
    manifest=task/"manifest.json"
    if not manifest.is_file():return None
    prepared=json.loads(manifest.read_text(encoding="utf-8"))["stages"].get("preprocessing",{})
    subject=prepared.get("subject")
    if not subject:return None
    path=reference_path.parent/(subject+"_landmark_affine.json")
    if not path.is_relative_to(task) or not path.is_file():return None
    record=json.loads(path.read_text(encoding="utf-8"))
    if Path(record.get("subject","")).resolve()!=reference_path:
        raise ValueError("个体仿射初始配准与当前 MRI 不匹配。")
    matrix=np.asarray(record.get("Affine"),dtype=float)
    if (matrix.shape!=(4,4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3],[0,0,0,1]) or np.linalg.det(matrix[:3,:3])<=0):
        raise ValueError("个体仿射初始配准矩阵无效。")
    return path,matrix


def subject_to_mni_lps(matrix):
    """ANTs pull transform: fixed subject LPS -> moving MNI LPS."""
    ras_to_lps=np.diag([-1.,-1.,1.])
    return ras_to_lps@matrix[:3,:3]@ras_to_lps,ras_to_lps@matrix[:3,3]


def save_atlas_record(task, record):
    from core.individual_pipeline import save_json

    manifest_path = task / 'manifest.json'
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        atlases = manifest.setdefault('atlases', {})
        if atlases.get(record['atlas_id']) != record:
            atlases[record['atlas_id']] = record
            save_json(manifest_path, manifest)


def register_atlas(config, progress=print):
    # A dedicated worker keeps ITK and its native library out of Qt/VTK.
    os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
    import ants
    from scipy.ndimage import binary_fill_holes, binary_dilation
    from nibabel.processing import resample_from_to
    task = Path(config['task_dir']).resolve()
    resources = Path(config['atlas_resources']).resolve()
    reference_path = Path(config['padded_t1']).resolve()
    tissue_path = Path(config['tissue_masks']).resolve()
    atlas_id = config.get('atlas_id', 'aal3')
    display_name = ATLAS_NAMES[atlas_id]
    template_mask = None
    voxel_labels_path = None
    if atlas_id == 'aal3':
        atlas_version = 'AAL3v1'
        atlas_path, template_path = resources/'AAL3v1_1mm.nii.gz', resources/'single_subj_T1.nii.gz'
        prior = preprocessing_affine(task, reference_path)
    else:
        atlas_version = '1.0'
        atlas_path = resources/'Morel_1mm.nii.gz'
        template_path = resources/'Morel_T1_1mm.nii.gz'
        template_mask = resources/'Morel_template_mask.nii.gz'
        voxel_labels_path = resources/'Morel_voxel_labels.json'
        prior = None
    labels = load_labels(resources, atlas_id)
    voxel_labels = json.loads(voxel_labels_path.read_text(encoding='utf-8')) if voxel_labels_path else {}
    membership_record = {'voxel_labels': voxel_labels}
    valid_codes = atlas_voxel_labels(membership_record, labels)
    paths = [reference_path, tissue_path, atlas_path, template_path, resources/(display_name + '_labels.json')]
    if template_mask is not None:
        paths.extend((template_mask, voxel_labels_path))
    if prior is not None:paths.append(prior[0])
    progress(f'正在核对 {display_name} 图谱与个体 MRI')
    signature = {'version':4,'initialization':'subject_to_mni_probability_template' if prior is not None else 'center_of_mass',
        'native_geometry':'full_affine_nibabel_ants_lps','registration_grid':'world_aligned_2mm','method':'ANTs SyNRA','ants_version':ants.__version__, 'seed':20261013,'threads':1,
        'registration_spacing_mm':2.0,'syn_iterations':[60,40,20],'aff_iterations':[1000,500,250,0],
        'interpolator':'nearestNeighbor','hashes':{p.name:file_hash(p) for p in paths}}
    key = hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    output = task/'atlas'/(atlas_id + '_' + key[:12]); output.mkdir(parents=True,exist_ok=True)
    record_path = output/'registration.json'; native_path = output/(display_name + '_native.nii.gz')
    reference = nib.load(reference_path)
    if record_path.is_file() and native_path.is_file():
        cached = json.loads(record_path.read_text(encoding='utf-8'))
        if cached.get('fingerprint') == key:
            validate_labels(nib.load(native_path), reference, valid_codes)
            save_atlas_record(task, cached)
            progress(f'复用已完成的 {display_name} 个体图谱'); return cached
    started = time.monotonic()
    tissue_img = nib.load(tissue_path)
    if tissue_img.shape != reference.shape or not np.allclose(tissue_img.affine,reference.affine,atol=1e-4):
        raise ValueError('组织分割与个体 MRI 的空间不一致。')
    brain = binary_fill_holes(np.isin(tissue_img.get_fdata(),[1,2,3]))
    if np.count_nonzero(brain) < 1000: raise ValueError('个体 MRI 的脑组织范围不足，无法配准图谱。')
    fixed_mask_path = output/'brain_mask.nii.gz'
    nib.save(nib.Nifti1Image(brain.astype(np.uint8),reference.affine),fixed_mask_path)
    template_img = nib.load(template_path)
    if template_mask is None:
        atlas_on_template = resample_from_to(nib.load(atlas_path),template_img,order=0).get_fdata()
        moving_mask = binary_dilation(binary_fill_holes(atlas_on_template>0),iterations=1)
        template_mask = output/'template_brain_mask.nii.gz'
        nib.save(nib.Nifti1Image(moving_mask.astype(np.uint8),template_img.affine),template_mask)
    fixed_full = nifti_to_ants(reference); moving_full = ants.image_read(str(template_path))
    # Optimizers and smoothing use an orthogonal world grid, even when the
    # training-compatible native padded header contains legacy shear.
    fixed,fixed_mask=atlas_working_grid(reference,brain)
    moving_mask_ants = ants.image_read(str(template_mask))
    moving = ants.iMath(moving_full * moving_mask_ants,'Normalize')
    fixed = ants.iMath(fixed * fixed_mask,'Normalize')
    initial_options={}
    if prior is not None:
        linear,offset=subject_to_mni_lps(prior[1])
        initial=ants.create_ants_transform(transform_type='AffineTransform',dimension=3,matrix=linear,offset=offset)
        initial_path=output/'subject_to_mni_initial.mat';ants.write_transform(initial,str(initial_path))
        initial_options['initial_transform']=[str(initial_path)]
        progress(f'复用电极定位阶段的个体仿射配准，初始化 {display_name} 脑区映射')
    progress(f'正在将 {display_name} 脑区映射到个体 MRI，首次处理需要一些时间')
    transform = ants.registration(fixed=fixed,moving=moving,type_of_transform='SyNRA',
        outprefix=str(output/'template_to_native_'), mask=fixed_mask,moving_mask=moving_mask_ants,
        mask_all_stages=True, aff_iterations=(1000,500,250,0),aff_shrink_factors=(6,4,2,1),
        aff_smoothing_sigmas=(3,2,1,0),reg_iterations=(60,40,20),syn_metric='CC',syn_sampling=2,
        random_seed=20261013,singleprecision=True,verbose=False,**initial_options)
    progress('正在生成个体脑区图谱并核对标签')
    warped = ants.apply_transforms(fixed=fixed_full,moving=ants.image_read(str(atlas_path)),
        transformlist=transform['fwdtransforms'],interpolator='nearestNeighbor',singleprecision=True)
    data = validate_labels(ants_on_reference(warped, reference), reference, valid_codes)
    covered = data>0
    overlap = float(np.count_nonzero(covered & brain)/max(1,np.count_nonzero(covered)))
    if overlap < .60: raise ValueError('图谱与脑组织的重合不足，请核对 MRI 和组织分割。')
    warped_t1 = ants.apply_transforms(fixed=fixed_full,moving=moving,
        transformlist=transform['fwdtransforms'],interpolator='linear',singleprecision=True)
    preview = output/'template_in_native.nii.gz'; nib.save(ants_on_reference(warped_t1,reference),preview)
    fixed_values = reference.get_fdata(dtype=np.float32)[brain]
    moving_values = nib.load(preview).get_fdata(dtype=np.float32)[brain]
    corr = float(np.corrcoef(fixed_values,moving_values)[0,1])
    if not np.isfinite(corr) or corr < .15: raise ValueError('图谱配准结果与 MRI 不匹配，请核对影像。')
    nib.save(nib.Nifti1Image(data,reference.affine),native_path)
    histogram = np.bincount(data[np.isin(tissue_img.get_fdata(),[1,2])],minlength=max(valid_codes)+1)
    native_histogram = np.bincount(data.ravel(), minlength=max(valid_codes)+1)
    codes_by_region = {str(k): atlas_voxel_labels(membership_record, [k]) for k in labels}
    counts = {k: int(histogram[codes].sum()) for k, codes in codes_by_region.items()}
    result = {'atlas_path':str(native_path),'atlas_id':atlas_id,'atlas_version':atlas_version,'display_name':display_name,
        'task_dir':str(task),'padded_t1':str(reference_path),'registration_path':str(record_path),
        'labels':{str(k):v for k,v in labels.items()},'gmwm_counts':counts,'fingerprint':key,'signature':signature,
        'native_counts':{k: int(native_histogram[codes].sum()) for k, codes in codes_by_region.items()},
        'fwdtransforms':transform['fwdtransforms'],'warped_template':str(preview),
        'quality':{'atlas_brain_overlap':overlap,'template_correlation':corr,'nonempty_regions':int(np.count_nonzero(list(counts.values()))),
            'elapsed_seconds':round(time.monotonic()-started,2)}}
    if voxel_labels:
        result['voxel_labels'] = voxel_labels
    record_temp = record_path.with_suffix('.tmp')
    record_temp.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8'); record_temp.replace(record_path)
    save_atlas_record(task, result)
    progress(f'个体 {display_name} 图谱已准备好，可以按脑区选择目标')
    return result
