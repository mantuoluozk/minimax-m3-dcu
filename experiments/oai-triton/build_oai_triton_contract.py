"""Separate HIP variant preserving the current Triton absmax floor."""
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('ROCM_HOME', '/opt/dtk')
os.environ.setdefault('PYTORCH_ROCM_ARCH', 'gfx938')
os.environ.setdefault('MAX_JOBS', '2')
import torch
from torch.utils.cpp_extension import load

root = Path(__file__).resolve().parent
target = root / 'native-oai-triton'
target.mkdir(exist_ok=True)
original = (root / 'native/swiglu_oai_fp8_quant_contiguous.cu').read_text()
assert original.count('const float scale = amax / 448.0f;') == 1
modified = original.replace('const float scale = amax / 448.0f;',
                          'const float scale = fmaxf(amax, 1.0e-10f) / 448.0f;')
modified = modified.replace('// Preserve the native scale=0/output=0 contract for all-zero rows.',
                          '// Preserve the current Triton absmax floor, including all-zero rows.')
(target / 'swiglu_oai_fp8_quant_contiguous.cu').write_text(modified)
(target / 'bindings.cpp').write_text('''#include <torch/extension.h>
namespace at::native {
std::vector<torch::Tensor> swiglu_oai_fp8_quant_contiguous(const torch::Tensor&, double, double, double);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
 m.def("swiglu_oai_fp8_quant_contiguous", &at::native::swiglu_oai_fp8_quant_contiguous);
}
''')
build = root / 'build-oai-triton'
build.mkdir(exist_ok=True)
module = load(name='m31_oai_triton_contract', sources=[str(target/'bindings.cpp'),
    str(target/'swiglu_oai_fp8_quant_contiguous.cu')], build_directory=str(build),
    extra_cflags=['-O3'], extra_cuda_cflags=['-O3','--offload-arch=gfx938'], verbose=True)
(root / 'oai-triton-build-manifest.json').write_text(json.dumps({
    'base_lightop_commit':'f162c8876e28694ef54b322722d9f67437b86814',
    'change':'absmax floor 1e-10, matching current Triton quantizer',
    'original_sha256':hashlib.sha256(original.encode()).hexdigest(),
    'modified_sha256':hashlib.sha256(modified.encode()).hexdigest(),
    'module':module.__file__, 'torch':torch.__version__}, indent=2))
print('OAI_TRITON_CONTRACT_BUILD_PASS', flush=True)
