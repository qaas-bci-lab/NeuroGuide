"""Explicit compatibility for the previously validated v2 numerical recipe.

Only known instrumentation/device-scheduling revisions may reuse that recipe.
Model, calibration, spatial, electrode and neural-network hashes stay strict.
"""
PREVIOUS_ADAPTER = '7325874885e24bdec816457ad811deed25c3f6a65782e82170ab92f10a54a222'
PREVIOUS_ONE_MM_ADAPTER='f4531fcd433aafb25170c1528165a32ec13f98c8755352702ce53efe22e46962'
# Only GPU preflight/fallback instrumentation changed; framework versions remain strict.
PREVIOUS_DEVICE_CHECK_ADAPTER='641833f91c12896960305e386cc46e30540b869192d880fe75ee9172a669421c'
PREVIOUS_INSTRUMENTED_SOURCES = {
    'segandelec_nnunet/pipeline.py': '890d219e995ec38d3e9ad735a00ebc25a46b5435f322798a2e9fc5b735c981d4'
}


def normalized(path):
    return str(path).replace('\\','/').casefold()


def signatures_match(saved, expected, allow_native_1mm=False):
    if saved.get('version')==2 and allow_native_1mm:
        saved=dict(saved,version=3,grid_policy='resampled_1mm_ras_padding',target_spacing_mm=1.0,resampling_order=1)
    if any(saved.get(key)!=value for key,value in expected.items() if key not in ('adapter_sha256','device','hashes')):
        return False
    return saved.get('adapter_sha256') in (expected.get('adapter_sha256'), PREVIOUS_ADAPTER,PREVIOUS_ONE_MM_ADAPTER,PREVIOUS_DEVICE_CHECK_ADAPTER)


def assets_match(saved, current):
    if len(saved)!=len(current): return False
    for relative,digest in current.items():
        old=[h for p,h in saved.items() if normalized(p).endswith('/'+normalized(relative))]
        permitted={digest}
        previous=PREVIOUS_INSTRUMENTED_SOURCES.get(normalized(relative))
        if previous: permitted.add(previous)
        if len(old)!=1 or old[0] not in permitted: return False
    return True
