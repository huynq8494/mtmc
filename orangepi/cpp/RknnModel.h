#pragma once
// Thin wrapper around one RKNN context (librknnrt). One instance must be used by one thread at a time.
#include <rknn_api.h>

#include <cstdint>
#include <string>
#include <vector>

struct RknnOutput {
    std::vector<float> data;
    std::vector<int> dims;
};

class RknnModel {
public:
    RknnModel() = default;
    ~RknnModel();
    RknnModel(const RknnModel&) = delete;
    RknnModel& operator=(const RknnModel&) = delete;

    // core_mask: 0 = auto, 1 = core0, 2 = core1, 4 = core2, 3 = core0+1, 7 = all (RK3588)
    void load(const std::string& path, int core_mask = 0);
    // rgb: contiguous uint8 HxWx3 (RGB, size input_h x input_w). Outputs are returned as float.
    void run(const uint8_t* rgb, std::vector<RknnOutput>& outs);

    int inputW() const { return in_w_; }
    int inputH() const { return in_h_; }

private:
    rknn_context ctx_ = 0;
    rknn_input_output_num io_{0, 0};
    std::vector<rknn_tensor_attr> out_attrs_;
    int in_w_ = 0, in_h_ = 0, in_c_ = 3;
};
