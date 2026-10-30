import gzip,json,tempfile,unittest
from pathlib import Path
import numpy as np
import nibabel as nib
from core.template_resources import template_files,template_path
from tools.package_resources import prepare_templates


class TemplateResourcesTests(unittest.TestCase):
    def test_shared_and_compressed_files_preserve_each_original_voxel_and_name(self):
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'original';out=Path(folder)/'packed'
            a=source/'ADHD'/'Target';b=source/'BD'/'Target';a.mkdir(parents=True);b.mkdir(parents=True)
            values=np.arange(120,dtype=np.float32).reshape(6,5,4);affine=np.diag([1.,2.,3.,1.])
            nib.save(nib.Nifti1Image(values,affine),a/'grey_field.nii')
            (b/'grey_field.nii').write_bytes((a/'grey_field.nii').read_bytes())
            report=prepare_templates(source,out)
            self.assertEqual(report['logical_files'],2);self.assertEqual(report['unique_payloads'],1)
            for disease in ('ADHD','BD'):
                directory=out/disease/'Target'
                self.assertTrue(directory.is_dir())
                self.assertEqual(template_files(out,directory),['grey_field.nii'])
                path=template_path(out,directory/'grey_field.nii');image=nib.load(path)
                np.testing.assert_array_equal(np.asarray(image.dataobj),values)
                np.testing.assert_array_equal(image.affine,affine)
                with gzip.open(path,'rb') as stream:self.assertEqual(stream.read(),(a/'grey_field.nii').read_bytes())
            self.assertEqual(template_path(out,out/'ADHD/Target/grey_field.nii'),template_path(out,out/'BD/Target/grey_field.nii'))

    def test_source_without_index_and_corrupt_mapping(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);directory=root/'Disease/Target';directory.mkdir(parents=True)
            file=directory/'field.nii';file.write_bytes(b'example')
            self.assertEqual(template_files(root,directory),['field.nii'])
            self.assertEqual(template_path(root,file),str(file))
            (root/'.resource_index.json').write_text(json.dumps({'files':{'Disease/Target/field.nii':'../outside.nii'}}))
            with self.assertRaises(ValueError):template_path(root,file)
