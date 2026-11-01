// Operators that must round like PyTorch: -O3 with --fmad=false -prec-div=true -prec-sqrt=true --ftz=false.
// All sources of this module are compiled as one translation unit.

#include "rope/rope_kernel.cu"
#include "pos_embed_interp/pos_embed_interp.cu"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rope", &wallx_cuda_rope::Rope);
  m.def("rope_full", &wallx_cuda_rope::RopeFull);
  m.def("rope_inplace", &wallx_cuda_rope::RopeInplace);
  m.def("rope_inplace_pack", &wallx_cuda_rope::RopeInplacePack);
  m.def("rope_inplace_pack_bwd", &wallx_cuda_rope::RopeInplacePackBackward);
  m.def("rope_bwd", &wallx_cuda_rope::RopeBackward);
  m.def("rope_full_bwd", &wallx_cuda_rope::RopeFullBackward);
  m.def("pos_embed_interp", &wallx_cuda_pos_embed_interp::PosEmbedInterp);
  m.def("pos_embed_interp_backward", &wallx_cuda_pos_embed_interp::PosEmbedInterpBackward);
}
