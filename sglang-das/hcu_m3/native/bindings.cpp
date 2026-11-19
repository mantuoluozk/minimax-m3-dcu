// Candidate-only bindings for HYGON-AI/lightop f162c8876e28694ef54b322722d9f67437b86814.
// Native sources retain their upstream notices; existing lightop.op is not replaced.
#include <torch/extension.h>
#include <vector>
namespace at::native {
torch::Tensor minimax_cp_fp8_pack(torch::Tensor,torch::Tensor,torch::Tensor,double,double,bool,bool);
void minimax_cp_cache_scatter(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor);
std::vector<torch::Tensor> minimax_qk_norm_rope(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,
    torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,double,int64_t,bool,int64_t);
std::vector<torch::Tensor> swiglu_oai_fp8_quant_contiguous(const torch::Tensor&,double,double,double);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.def("minimax_cp_fp8_pack",&at::native::minimax_cp_fp8_pack);
    m.def("minimax_cp_cache_scatter",&at::native::minimax_cp_cache_scatter);
    m.def("minimax_qk_norm_rope",&at::native::minimax_qk_norm_rope);
    m.def("swiglu_oai_fp8_quant_contiguous",&at::native::swiglu_oai_fp8_quant_contiguous);
}
