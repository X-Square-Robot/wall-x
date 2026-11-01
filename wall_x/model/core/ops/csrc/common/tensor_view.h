// Minimal tensor view used by the wall_x CUDA operators.
//
// The kernels in this directory take tensors through a small DLPack-style view:
// raw data pointer, shape/strides, dtype code/bits and device. This header builds
// that view from an at::Tensor exactly the way PyTorch exports a tensor to DLPack
// (at::toDLPack), so every shape/stride/dtype check sees the same values it would
// see through any DLPack consumer.
#pragma once

#include <ATen/DLConvertor.h>
#include <ATen/dlpack.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <memory>
#include <stdexcept>
#include <string>

// DLPack dtypes compare by code, bits and lanes.
inline bool operator==(const DLDataType& lhs, const DLDataType& rhs) {
  return lhs.code == rhs.code && lhs.bits == rhs.bits && lhs.lanes == rhs.lanes;
}
inline bool operator!=(const DLDataType& lhs, const DLDataType& rhs) { return !(lhs == rhs); }

namespace wallx_cuda {

class TensorView {
 public:
  TensorView() : tensor_{} {}  // empty; only used as a pybind11 placeholder
  TensorView(const at::Tensor& tensor)  // NOLINT(google-explicit-constructor)
      : owner_(at::toDLPack(tensor), [](DLManagedTensor* m) { if (m && m->deleter) m->deleter(m); }),
        tensor_(owner_->dl_tensor) {}

  void* data_ptr() const { return tensor_.data; }
  DLDevice device() const { return tensor_.device; }
  int32_t ndim() const { return tensor_.ndim; }
  int32_t dim() const { return tensor_.ndim; }
  DLDataType dtype() const { return tensor_.dtype; }

  int64_t size(int64_t idx) const { return tensor_.shape[Adjust(idx)]; }

  int64_t stride(int64_t idx) const {
    if (tensor_.strides == nullptr) {
      int64_t s = 1;
      for (int32_t k = tensor_.ndim - 1; k > Adjust(idx); --k) s *= tensor_.shape[k];
      return s;
    }
    return tensor_.strides[Adjust(idx)];
  }

  int64_t numel() const {
    int64_t n = 1;
    for (int32_t k = 0; k < tensor_.ndim; ++k) n *= tensor_.shape[k];
    return n;
  }

  // Same rule as the DLPack contiguity check: size-1 dimensions are skipped.
  bool IsContiguous() const {
    if (tensor_.strides == nullptr) return true;
    int64_t expected = 1;
    for (int32_t i = tensor_.ndim; i != 0; --i) {
      int32_t k = i - 1;
      if (tensor_.shape[k] == 1) continue;
      if (tensor_.strides[k] != expected) return false;
      expected *= tensor_.shape[k];
    }
    return true;
  }

 private:
  int64_t Adjust(int64_t idx) const {
    int64_t adjusted = idx >= 0 ? idx : tensor_.ndim + idx;
    if (adjusted < 0 || adjusted >= tensor_.ndim) {
      throw std::out_of_range("Index " + std::to_string(idx) + " out of bounds for tensor with " +
                              std::to_string(tensor_.ndim) + " dimensions");
    }
    return adjusted;
  }

  std::shared_ptr<DLManagedTensor> owner_;
  DLTensor tensor_;
};

class OptionalTensorView {
 public:
  OptionalTensorView() = default;
  OptionalTensorView(const c10::optional<at::Tensor>& tensor) {  // NOLINT(google-explicit-constructor)
    if (tensor.has_value() && tensor->defined()) value_ = std::make_shared<TensorView>(*tensor);
  }
  bool has_value() const { return value_ != nullptr; }
  const TensorView& value() const {
    if (!value_) throw std::runtime_error("optional tensor has no value");
    return *value_;
  }

 private:
  std::shared_ptr<TensorView> value_;
};

// Current PyTorch stream on the given device (the stream the caller is on).
inline void* CurrentStream(int32_t /*device_type*/, int32_t device_id) {
  return at::cuda::getCurrentCUDAStream(device_id).stream();
}

}  // namespace wallx_cuda

namespace pybind11 {
namespace detail {

template <>
struct type_caster<wallx_cuda::TensorView> {
 public:
  PYBIND11_TYPE_CASTER(wallx_cuda::TensorView, _("torch.Tensor"));
  bool load(handle src, bool convert) {
    make_caster<at::Tensor> inner;
    if (!inner.load(src, convert)) return false;
    value = wallx_cuda::TensorView(cast_op<at::Tensor>(inner));
    return true;
  }
};

template <>
struct type_caster<wallx_cuda::OptionalTensorView> {
 public:
  PYBIND11_TYPE_CASTER(wallx_cuda::OptionalTensorView, _("Optional[torch.Tensor]"));
  bool load(handle src, bool convert) {
    if (src.is_none()) {
      value = wallx_cuda::OptionalTensorView();
      return true;
    }
    make_caster<at::Tensor> inner;
    if (!inner.load(src, convert)) return false;
    value = wallx_cuda::OptionalTensorView(c10::optional<at::Tensor>(cast_op<at::Tensor>(inner)));
    return true;
  }
};

}  // namespace detail
}  // namespace pybind11
