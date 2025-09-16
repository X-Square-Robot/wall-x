#include "dual_asym_grouped_gemm.h"
#include "permute.h"
#include "rope.h"
#include "rope_index.h"
#include "rot_pos.h"
#include "window_index.h"

#include <torch/extension.h>


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("asym_dual_gmm", &AsymmetricDualExpertGemm, "Asymmetric Dual Expert Grouped GEMM.");
  m.def("permute", &moe_permute_topK_op, "Token permutation kernel");
  m.def("unpermute", &moe_recover_topK_op, "Token un-permutation kernel");
  m.def("unpermute_bwd", &moe_recover_topK_bwd_op, "Token un-permutation backward kernel");
  m.def("rope", &launch_multimodal_rope_forward, "Multimodal RoPE forward kernel");
  m.def("rope_bwd", &launch_multimodal_rope_backward, "Multimodal RoPE backward kernel");
  m.def("rope_index", &get_rope_index, "Get RoPE index kernel");
  m.def("rot_pos_emb", &fused_rot_pos_emb_cuda, "Fused Rotary Position Embedding kernel");
  m.def("get_window_index", &get_window_index_cuda, "Get window index kernel");
}
