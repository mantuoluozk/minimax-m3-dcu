import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('PYTORCH_ROCM_ARCH','gfx938')
os.environ.setdefault('MAX_JOBS','2')
from torch.utils.cpp_extension import load
import torch

source_dir = Path(__file__).resolve().parent
root = source_dir.parents[1]
build=root/'build'
build.mkdir(exist_ok=True)
sources=[source_dir/'native'/name for name in ['bindings.cpp','minimax_cp_fp8_pack.cu',
    'minimax_cp_cache_scatter.cu','minimax_qk_norm_rope.cu','swiglu_oai_fp8_quant_contiguous.cu']]
module=load(name='m31_minimax_ops',sources=[str(p) for p in sources],build_directory=str(build),
    extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--offload-arch=gfx938'],verbose=True)
manifest={'lightop_commit':'f162c8876e28694ef54b322722d9f67437b86814','torch':torch.__version__,
    'hip':torch.version.hip,'arch':'gfx938','module':module.__file__,
    'sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    'module_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()}
(root/'native-build-manifest.json').write_text(json.dumps(manifest,indent=2))
print('M31_CANDIDATE_NATIVE_BUILD_PASS',json.dumps(manifest),flush=True)
