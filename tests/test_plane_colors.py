import unittest
import numpy as np
from ui.plane_colors import slice_rgba

class PlaneColorTests(unittest.TestCase):
    def test_visible_field_threshold_opacity_and_saturation(self):
        base=np.ones((3,3))*.5
        data=np.ones((3,3));data[0,0]=0;data[0,1]=np.nan
        field={'data':data,'min':0.,'max':.9,'cmap':'jet','alpha':1.}
        colored=slice_rgba(base,[field],0,1)
        np.testing.assert_array_equal(colored[0,0,:3],[128]*3)
        np.testing.assert_array_equal(colored[0,1,:3],[128]*3)
        self.assertGreater(colored[1,1,0],colored[1,1,1])
        field['alpha']=0
        np.testing.assert_array_equal(slice_rgba(base,[field],0,1)[...,:3],np.full((3,3,3),128))
        self.assertTrue((colored[...,3]==255).all())

    def test_roi_outline_preserves_inside_field(self):
        base=np.zeros((5,5));roi=np.zeros_like(base);roi[1:4,1:4]=1
        result=slice_rgba(base,[{'data':roi,'is_roi':True}],0,1)
        self.assertEqual(result[2,2,0],0)
        self.assertEqual(result[1,1,0],255)
        self.assertEqual(result[0,0,3],0)
