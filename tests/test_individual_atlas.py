import unittest
import numpy as np
import nibabel as nib
from core.atlas import validate_labels,region_name

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
