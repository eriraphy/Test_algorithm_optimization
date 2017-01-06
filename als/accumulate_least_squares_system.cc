//
// Uber, Inc. (c) 2016
//

#include <Eigen/Dense>

#include <exception>
#include <iostream>
#include <random>
#include <string>
#include <vector>

struct match_result
{
    uint32_t index;
    float error;

    match_result(uint32_t ind, float err) : index(ind), error(err)
    {}

    // Comparison operator based on error field
    bool operator<(const match_result &other) const
    {
        return error < other.error;
    }
};


// Builds up a system matrix based on the errors; does not solve it
void accumulate_system(const Eigen::Matrix<float, 3, Eigen::Dynamic> &readings,
                       const Eigen::Matrix<float, 3, Eigen::Dynamic> &points,
                       const Eigen::Matrix<float, 3, Eigen::Dynamic> &normals,
                       const std::vector<std::vector<match_result>> &matches,
                       std::vector<std::vector<float>> &weights,
                       double &residual,
                       Eigen::Matrix<double, 6, 6> &H,
                       Eigen::Matrix<double, 6, 1> &Z)
{
    if ((points.size() != normals.size()))
    {
        throw std::runtime_error(std::string("Parameter sizes do not match"));
    }

    using mat6x6 = Eigen::Matrix<double, 6, 6>;
    using vec6 = Eigen::Matrix<double, 6, 1>;
    using vec3 = Eigen::Matrix<double, 3, 1>;

    const size_t num_readings = static_cast<size_t>(readings.cols());

    weights.resize(num_readings);
    float sum_of_weights;
    constexpr uint32_t MAX_ITERS = 10u;
    for (uint32_t iter = 0; iter <= MAX_ITERS; ++iter)
    {
        // nonsense random 'transformation'
        const Eigen::Matrix<double, 3, 4> update_mat = Eigen::MatrixXd::Random(3, 4);

        const bool compute_update = (iter < MAX_ITERS);

        mat6x6 H = mat6x6::Zero();
        vec6 Z = vec6::Zero();
        residual = 0;
        sum_of_weights = 0;
        for (size_t i = 0; i < num_readings; ++i)
        {
            const vec3 pt = readings.col(i).cast<double>();
            const vec3 p = (0 == iter) ? pt : (update_mat * Eigen::Vector4d(pt(0), pt(1), pt(2), 1.0));

            const size_t num_matches_i = matches[i].size();
            weights[i].resize(num_matches_i);
            for (size_t m = 0; m < num_matches_i; ++m)
            {
                const uint32_t ref_id = matches[i][m].index;
                const vec3 &n = normals.col(ref_id).cast<double>();
                const vec3 &q = points.col(ref_id).cast<double>();
                const double err = n.dot(q - p);

                constexpr double K2 = 0.1 * 0.1;
                const double err2 = (err * err);
                const double cost = (err2 < K2) ? err2 : 2.0 * std::sqrt(err2 * K2) - K2;
                residual += cost;

                const float weight = (err2 < K2) ? 1.0f : static_cast<float>(std::sqrt(K2 / err2));
                weights[i][m] = weight;

                if (0 == weight)
                {
                    continue;
                }
                sum_of_weights += weight;

                if (compute_update)
                {
                    vec6 JT;
                    JT.head<3>() = p.cross(n);
                    JT.tail<3>() = n;
                    const vec6 JTw = JT * weight;
                    H += JTw * JT.transpose();
                    Z += JTw * err;
                }
            }
        }
    }
}


int32_t main(int32_t argc, char *argv[])
{
    constexpr double SCALE = 100.0;
    constexpr int32_t NUM_READINGS = 20000;
    const Eigen::Matrix<float, 3, Eigen::Dynamic> readings = Eigen::MatrixXf::Random(3, NUM_READINGS) * SCALE;

    constexpr int32_t NUM_REFS = 200000;
    const Eigen::Matrix<float, 3, Eigen::Dynamic> points = Eigen::MatrixXf::Random(3, NUM_REFS) * SCALE;
    Eigen::Matrix<float, 3, Eigen::Dynamic> normals = Eigen::MatrixXf::Random(3, NUM_REFS);
    normals.colwise().normalize();

    std::mt19937 gen(47u);
    std::uniform_int_distribution<uint32_t> dist(0, NUM_REFS);
    constexpr uint32_t NUM_NN = 3u;
    std::vector<std::vector<match_result>> matches(NUM_READINGS);
    for (auto &m : matches)
    {
        for (size_t i = 0; i < NUM_NN; ++i)
        {
            const uint32_t random_index = dist(gen);
            m.emplace_back(random_index, 2.0);
        }
    }

    std::vector<std::vector<float>> weights;
    Eigen::Matrix<double, 6, 6> H;
    Eigen::Matrix<double, 6, 1> Z;
    constexpr uint32_t NUM_RUNS = 1000u;
    for (uint32_t r = 0; r < NUM_RUNS; ++r)
    {
        double residual;
        accumulate_system(readings, points, normals, matches, weights, residual, H, Z);
        std::cout.precision(10);
        std::cout << "residual = " << residual << std::endl;
    }

    return 0;
}
