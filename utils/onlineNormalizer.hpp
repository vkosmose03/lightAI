/*
 *
 *      Created by vkosmose03
 *
 *      Part of lightAI lib that provide normalization and denormalization
 *      
 *      path: lightAI/utils/onlineNormalizer.hpp
 */

#ifndef __ONLINE_NORMALIZER_HPP
#define __ONLINE_NORMALIZER_HPP

#include <cmath>
#include <cstdint>

namespace lightAI::utils {

class OnlineNormalizer {
 public:
    OnlineNormalizer() : n_(0), mean_(0.0), M_(0.0) {}

    void update(double x) {
        ++n_;
        double delta  = x - mean_;
        mean_ += delta / static_cast<double>(n_);
        double delta2 = x - mean_;
        M_    += delta * delta2;
    }

    double mean()   const { return mean_; }
    double stddev() const {
        if (n_ < 2) return 1.0;
        double s = std::sqrt(M_ / static_cast<double>(n_));
        return (s < 1e-8) ? 1.0 : s;
    }
    int64_t count() const { return n_; }

    double normalize(double x)   const { return (x - mean()) / stddev(); }

    double denormalize(double y) const { return y * stddev() + mean(); }

    int64_t n_;
    double  mean_;
    double  M_;
};

} // namespace lightAI::utils

#endif
