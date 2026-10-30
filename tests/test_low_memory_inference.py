import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'BrainLFNet'))
from brainlfnet import BrainLFNet
from core.low_memory_inference import LowMemoryInference

class LowMemoryTests(unittest.TestCase):
    def test_same_full_volume_result_odd_shapes_and_identity_shortcut(self):
        torch.manual_seed(6)
        for base in (2,8):
            model=BrainLFNet(base_channels=base).eval()
            with torch.no_grad():
                for block in model.modules():
                    if isinstance(block,torch.nn.GroupNorm):
                        block.weight.uniform_(.5,1.5);block.bias.uniform_(-.2,.2)
            volume=torch.randn(1,8,17,19,21);saved=volume.clone()
            query=torch.tensor([[[0,0,0],[8,9,10],[16,18,20]]])
            with torch.inference_mode():
                reference=model(volume,query);low=LowMemoryInference(model)(volume,query)
            torch.testing.assert_close(low,reference,rtol=1e-5,atol=1e-6)
            torch.testing.assert_close(volume,saved,rtol=0,atol=0)

    def test_not_available_for_training_or_gradients(self):
        model=BrainLFNet(base_channels=2)
        with self.assertRaises(ValueError):LowMemoryInference(model)
        executor=LowMemoryInference(model.eval())
        with self.assertRaises(RuntimeError):executor.dense(torch.zeros(1,8,16,16,16))
