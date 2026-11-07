#include "RknnModel.h"

#include <cstring>
#include <fstream>
#include <stdexcept>

RknnModel::~RknnModel() {
    if (ctx_ != 0) rknn_destroy(ctx_);
}

void RknnModel::load(const std::string& path, int core_mask) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) throw std::runtime_error("cannot open RKNN model: " + path);
    const std::streamsize size = f.tellg();
    if (size <= 0) throw std::runtime_error("empty RKNN model: " + path);
    std::vector<char> buf(static_cast<size_t>(size));
    f.seekg(0);
    f.read(buf.data(), size);

    int ret = rknn_init(&ctx_, buf.data(), static_cast<uint32_t>(buf.size()), 0, nullptr);
    if (ret != RKNN_SUCC) throw std::runtime_error("rknn_init failed (" + std::to_string(ret) + "): " + path);
    if (core_mask != 0) {
        ret = rknn_set_core_mask(ctx_, static_cast<rknn_core_mask>(core_mask));
        if (ret != RKNN_SUCC) throw std::runtime_error("rknn_set_core_mask failed: " + std::to_string(ret));
    }

    ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_, sizeof(io_));
    if (ret != RKNN_SUCC || io_.n_input < 1 || io_.n_output < 1)
        throw std::runtime_error("rknn_query IN_OUT_NUM failed: " + path);

    rknn_tensor_attr in{};
    in.index = 0;
    ret = rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &in, sizeof(in));
    if (ret != RKNN_SUCC) throw std::runtime_error("rknn_query INPUT_ATTR failed");
    if (in.fmt == RKNN_TENSOR_NCHW) {
        in_c_ = in.dims[1]; in_h_ = in.dims[2]; in_w_ = in.dims[3];
    } else {
        in_h_ = in.dims[1]; in_w_ = in.dims[2]; in_c_ = in.dims[3];
    }

    out_attrs_.assign(io_.n_output, rknn_tensor_attr{});
    for (uint32_t i = 0; i < io_.n_output; ++i) {
        out_attrs_[i].index = i;
        ret = rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &out_attrs_[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) throw std::runtime_error("rknn_query OUTPUT_ATTR failed");
    }
}

void RknnModel::run(const uint8_t* rgb, std::vector<RknnOutput>& outs) {
    rknn_input in;
    std::memset(&in, 0, sizeof(in));
    in.index = 0;
    in.type = RKNN_TENSOR_UINT8;
    in.fmt = RKNN_TENSOR_NHWC;
    in.size = static_cast<uint32_t>(in_w_ * in_h_ * in_c_);
    in.buf = const_cast<uint8_t*>(rgb);
    in.pass_through = 0;
    int ret = rknn_inputs_set(ctx_, 1, &in);
    if (ret != RKNN_SUCC) throw std::runtime_error("rknn_inputs_set failed: " + std::to_string(ret));

    ret = rknn_run(ctx_, nullptr);
    if (ret != RKNN_SUCC) throw std::runtime_error("rknn_run failed: " + std::to_string(ret));

    std::vector<rknn_output> o(io_.n_output);
    std::memset(o.data(), 0, sizeof(rknn_output) * o.size());
    for (uint32_t i = 0; i < io_.n_output; ++i) {
        o[i].index = i;
        o[i].want_float = 1;  // runtime dequantizes int8 outputs and fixes the layout
    }
    ret = rknn_outputs_get(ctx_, io_.n_output, o.data(), nullptr);
    if (ret != RKNN_SUCC) throw std::runtime_error("rknn_outputs_get failed: " + std::to_string(ret));

    outs.resize(io_.n_output);
    for (uint32_t i = 0; i < io_.n_output; ++i) {
        const float* p = static_cast<const float*>(o[i].buf);
        outs[i].data.assign(p, p + o[i].size / sizeof(float));
        outs[i].dims.assign(out_attrs_[i].dims, out_attrs_[i].dims + out_attrs_[i].n_dims);
    }
    rknn_outputs_release(ctx_, io_.n_output, o.data());
}
