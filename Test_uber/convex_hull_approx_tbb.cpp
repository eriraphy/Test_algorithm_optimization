//
// Uber, Inc. (c) 2016
//

#include <algorithm>
#include <array>
#include <iostream>
#include <random>
#include <vector>
#include <string>
#include <tuple> // For std::tie
#include <iterator> // For global begin() and end()


#include <stdint.h>
#include <time.h>

#define NOMINMAX
#include <windows.h>


#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
using namespace tbb;

// Functions from http://stackoverflow.com/questions/17432502/how-can-i-measure-cpu-time-and-wall-clock-time-on-both-linux-windows
// to measure wall time and cpu time
double get_cpu_time(){
	FILETIME a, b, c, d;
	if (GetProcessTimes(GetCurrentProcess(), &a, &b, &c, &d) != 0) {
		return
			(double)(d.dwLowDateTime | ((unsigned long long)d.dwHighDateTime << 32))*0.0000001;
	}
	else {
		return 0;
	}
}
double get_wall_time(){
    return (double)clock() / CLOCKS_PER_SEC;
}

// Timer utility
class measure_time_and_print
{
public:
    measure_time_and_print(std::string name)
        : customized_name(std::move(name)),start_wall_time(get_wall_time()), start_cpu_time(get_cpu_time())
    {
    }

    ~measure_time_and_print()
    {
        std::cout << "Wall time for " << customized_name << " is " << get_wall_time() - start_wall_time << std::endl;
        std::cout << "CPU time for " << customized_name << " is " << get_cpu_time() - start_cpu_time << std::endl << std::endl;
    }
private:
    const double start_wall_time;
    const double start_cpu_time;
    const std::string customized_name;
};


/// Simple 2D point structure with initializer. 
struct point2d
{
	double x = 0;
	double y = 0;

	point2d() = default;
    point2d(double initial_x, double initial_y)
    {
        x = initial_x;
        y = initial_y;
    }
};

/// Histogram bin structure that holds highest and lowest point for that x column. 
/// Has a used flag to indicate if the low/high points are populated.
/// Note high and low may be the same point if only one point fell into it. 
struct point_bin
{
    point_bin() : 
		low(std::numeric_limits<double>::max(), std::numeric_limits<double>::max()),
		high(std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest())
    {
    }
    point2d low;
    point2d high;
    //bool used = false;
};

class convex_hull_approx_test
{
public:
    static constexpr int32_t NUM_POINTS = 50000;
    static constexpr double DISCRETIZATION_SIZE = 200;
    convex_hull_approx_test() = default;
    ~convex_hull_approx_test() = default;



    /// Test entry point
    static void run_test();

private:
    void generate_points(int32_t seed, std::vector<point2d> &points);
    
    std::mt19937 random_generator;
};

void convex_hull_approx_test::generate_points(int32_t seed, std::vector<point2d> &points)
{
    random_generator.seed(seed);
    std::uniform_real_distribution<double> dist_real_values(0, 10.0);
    for (auto &point : points)
    {
        point.x = dist_real_values(random_generator);
        point.y = dist_real_values(random_generator);
    }
}

bool pointsxcompare(point2d & a, point2d & b) { return a.x < b.x; }

class bincomp {
	std::vector<std::vector<point2d>> &set_of_points;
	double DISCRETIZATION_SIZE;
	//std::vector<point2d> &approx_convex_hull;

public:
	void operator() (const blocked_range<size_t> &r) const {
		for (size_t j = r.begin(); j < r.end(); j++) {

			std::vector<point2d > approx_convex_hull;
			auto &points = set_of_points[j];
			auto minmax_x = std::minmax_element(std::begin(points), std::end(points), pointsxcompare);

			// Histogram points by x-coordinate, saving the highest and lowest points (by y-coordinate) in each bin. 
			const double range_x = minmax_x.second->x - minmax_x.first->x;
			const int32_t num_bins = std::floor(range_x * DISCRETIZATION_SIZE) + 1;
			std::vector<point_bin> bins(num_bins);

			for (int i = 0; i < points.size(); i++) {
				const auto &point = points[i];

				const int64_t bin_index = (point.x - minmax_x.first->x) * DISCRETIZATION_SIZE;

				auto &bin = bins[bin_index];

				if (point.y < bin.low.y)
				{
					bin.low = point;
				}
				else if (point.y > bin.high.y)
				{
					bin.high = point;
				}
			}

			// Construct the lower convex polygon. 
			approx_convex_hull.resize(num_bins * 2);
			int32_t index = 0;
			for (size_t i = 0; i < bins.size(); ++i)
			{

				auto &bin = bins[i];
				if (bin.low.x == std::numeric_limits<double>::max() && bin.high.x == std::numeric_limits<double>::lowest()) {
					continue;
				}
				else if (bin.low.x == std::numeric_limits<double>::max()) {
					bin.low = bin.high;
				}
				else if (bin.low.x == std::numeric_limits<double>::lowest()) {
					bin.high = bin.low;
				}
				const auto &p_low = bin.low;

				while (index >= 2)
				{
					const auto &pa = approx_convex_hull[index - 2];
					const auto &pb = approx_convex_hull[index - 1];
					const double dist = ((pb.x - pa.x) * (p_low.y - pa.y) - (pb.y - pa.y) * (p_low.x - pa.x));
					if (dist > 0)
					{
						break;
					}
					else
					{
						// pb isn't actually on the polygon, remove. 
						--index;
					}
				}
				approx_convex_hull[index] = p_low;
				++index;
			}

			// If upper and lower bins are the same, don't replicate
			const auto &last_bin = bins[bins.size() - 1];
			if (last_bin.low.x != last_bin.high.x && last_bin.low.y != last_bin.high.y)
			{
				approx_convex_hull[index] = last_bin.high;
				++index;
			}

			// Now extract the upper hull. 
			const int32_t start_index = index;
			for (int32_t i = static_cast<int32_t>(bins.size()) - 2; i >= 0; --i)
			{
				auto &bin = bins[i];
				if (bin.low.x == std::numeric_limits<double>::max() && bin.high.x == std::numeric_limits<double>::lowest()) {
					continue;
				}
				else if (bin.low.x == std::numeric_limits<double>::max()) {
					bin.low = bin.high;
				}
				else if (bin.low.x == std::numeric_limits<double>::lowest()) {
					bin.high = bin.low;
				}
				const auto &p_high = bin.high;

				while (index - start_index >= 1)
				{
					// http://stackoverflow.com/questions/1560492/how-to-tell-whether-a-point-is-to-the-right-or-left-side-of-a-line
					const auto &pa = approx_convex_hull[index - 2];
					const auto &pb = approx_convex_hull[index - 1];
					const double dist = ((pb.x - pa.x) * (p_high.y - pa.y) - (pb.y - pa.y) * (p_high.x - pa.x));
					if (dist > 0)
					{
						break;
					}
					else
					{
						--index;
					}
				}
				approx_convex_hull[index] = p_high;
				++index;
			}

			// If start and end points are the same, don't replicate
			const auto &last_hull_point = approx_convex_hull[index - 1];
			const auto &first_hull_point = approx_convex_hull[0];
			if (first_hull_point.x == last_hull_point.x && first_hull_point.y == last_hull_point.y)
			{
				--index;
			}

			// This is how many points we should have. 
			approx_convex_hull.resize(index);

		}
	}
	bincomp(std::vector<std::vector<point2d>> &p, double s/*, std::vector<point2d> &ch*/) :
		set_of_points(p), DISCRETIZATION_SIZE(s)/*, approx_convex_hull(ch)*/ {}
};

void convex_hull_approx_test::run_test()
{
	// Generate a bunch of uniformly random 2D points. 
	constexpr int32_t NUM_RUNS = 3000;
	std::vector<std::vector<point2d>> set_of_points;
	convex_hull_approx_test test;

	//tbb::task_scheduler_init init(4);

	for (int32_t i = 1; i <= NUM_RUNS; ++i)
	{
		std::vector<point2d> points(i * NUM_POINTS / NUM_RUNS);

		test.generate_points(i, points);
		set_of_points.push_back(points);
	}
	

	// This is what we want to optimize: extracting an approximate convex polygon
	// for each set of points. 
	{
		measure_time_and_print measure_time("convex_hull_approx_test");

			std::vector<std::vector<point2d>> approx_convex_hull_set;
			parallel_for(blocked_range<size_t>(0, set_of_points.size()), bincomp(set_of_points, DISCRETIZATION_SIZE));

	}
}

int main()
{
    convex_hull_approx_test::run_test();
	//system("pause");
}
