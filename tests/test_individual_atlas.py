from pathlib import Path
import unittest
import numpy as np
import nibabel as nib
from core.atlas import load_labels, region_name, validate_labels

class AtlasTests(unittest.TestCase):
    def test_label_integrity_and_native_grid(self):
        reference=nib.Nifti1Image(np.zeros((5,5,5)),np.eye(4))
        good=nib.Nifti1Image(np.full((5,5,5),41,dtype=np.int16),np.eye(4))
        self.assertEqual(np.unique(validate_labels(good,reference,[41,42])).tolist(),[41])
        bad=nib.Nifti1Image(np.full((5,5,5),41.5),np.eye(4))
        with self.assertRaises(ValueError): validate_labels(bad,reference,[41,42])
        shift=np.eye(4); shift[0,3]=10
        with self.assertRaises(ValueError): validate_labels(nib.Nifti1Image(good.get_fdata(),shift),reference,[41,42])
        with self.assertRaises(ValueError): validate_labels(good,reference,[1,2])

    def test_bilingual_region_names_and_laterality(self):
        self.assertEqual(region_name('Hippocampus_L'),'左海马')
        self.assertEqual(region_name('Hippocampus_R'),'右海马')
        self.assertEqual(region_name('UnknownRegion'),'UnknownRegion')

    def test_every_morel_label_has_a_distinct_chinese_name(self):
        resources = Path(__file__).resolve().parents[1] / 'assets/atlases'
        if not (resources / 'Morel_labels.json').is_file():
            self.skipTest('Morel atlas resources are not installed')
        labels = load_labels(resources, 'morel').values()
        translations = set()
        for label in labels:
            with self.subTest(label=label):
                translated = region_name(label)
                self.assertTrue(translated.startswith('左' if label.endswith('_L') else '右'))
                self.assertRegex(translated[1:], r'^[\u4e00-\u9fff]+$')
                self.assertNotIn(translated, translations)
                translations.add(translated)
        self.assertEqual(len(translations), 78)
        self.assertEqual(region_name('CM_L'), '左丘脑中央中核')
        self.assertEqual(region_name('CeM_L'), '左丘脑中央内侧核')
        self.assertEqual(region_name('VLpd_R'), '右丘脑腹外侧后核背侧部')
        self.assertEqual(region_name('VLpv_R'), '右丘脑腹外侧后核腹侧部')
