"""Full-volume inference with CPU skip storage and owned residual temporaries.

No spatial patches: GroupNorm and the source-conditioned context retain the
entire volume. Weights/layout/units are unchanged; CUDA activations use FP16.
"""
import torch
import torch.nn.functional as F


class LowMemoryInference:
    def __init__(self,model):
        if model.training:raise ValueError('Low-memory execution requires an evaluation model')
        self.model=model

    @staticmethod
    def residual(module,owned):
        # The caller relinquishes the concatenation so it can be freed after
        # the first convolution. An Identity shortcut deliberately retains it.
        x=owned.pop()
        residual=module.shortcut(x)
        value=module.conv1(x);del x
        normalized=module.norm1(value);del value
        torch.ops.aten.gelu_.default(normalized)
        value=module.conv2(normalized);del normalized
        normalized=module.norm2(value);del value
        normalized.add_(residual);del residual
        return torch.ops.aten.gelu_.default(normalized)

    def dense(self,volume):
        if torch.is_grad_enabled():raise RuntimeError('Low-memory execution is inference-only')
        model=self.model
        sources=[volume[:,6:8]]
        for _ in range(4):sources.append(model.electrode_pool(sources[-1]))
        skips=[]
        x=self.residual(model.encoders[0],[volume])
        for level in range(1,5):
            skips.append(x.to('cpu'))
            pooled=model.pool(x);del x
            owned=[torch.cat([pooled,sources[level]],1)];del pooled
            x=self.residual(model.encoders[level],owned)
        x=model.context(x)
        for decoder,source in zip(model.decoders,reversed(sources[:-1])):
            skip=skips.pop().to(volume.device)
            up=decoder.upsample(x);del x
            if up.shape[-3:]!=skip.shape[-3:]:
                up=F.interpolate(up,size=skip.shape[-3:],mode='trilinear',align_corners=False)
            if source.shape[-3:]!=skip.shape[-3:]:
                source=F.adaptive_max_pool3d(source,output_size=skip.shape[-3:])
            owned=[torch.cat([up,skip,source],1)];del up,skip
            x=self.residual(decoder.refine,owned)
        return model.output_projection(x)

    def __call__(self,volume,query):
        return self.model.sample_queries(self.dense(volume),query)
