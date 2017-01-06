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
#include <windows.h>

#define _DBL_MAX DBL_MAX
#define _DBL_MIN -DBL_MAX

// Functions from http://stackoverflow.com/questions/17432502/how-can-i-measure-cpu-time-and-wall-clock-time-on-both-linux-windows
// to measure wall time and cpu time
double get_cpu_time() {
	FILETIME a, b, c, d;
	if (GetProcessTimes(GetCurrentProcess(), &a, &b, &c, &d) != 0) {
		return
			(double)(d.dwLowDateTime | ((unsigned long long)d.dwHighDateTime << 32))*0.0000001;
	}
	else {
		return 0;
	}
}
double get_wall_time() {
	return (double)clock() / CLOCKS_PER_SEC;
}

// Timer utility
class measure_time_and_print
{
public:
	measure_time_and_print(std::string name)
		: customized_name(std::move(name)), start_wall_time(get_wall_time()), start_cpu_time(get_cpu_time())
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
	point2d() = default;
	point2d(double initial_x, double initial_y)
	{
		x = initial_x;
		y = initial_y;
	}

	double x = 0;
	double y = 0;
};

/// Histogram bin structure that holds highest and lowest point for that x column. 
/// Has a used flag to indicate if the low/high points are populated.
/// Note high and low may be the same point if only one point fell into it. 
struct point_bin
{
	point_bin() :
		low(_DBL_MAX, _DBL_MAX),
		high(_DBL_MIN, _DBL_MIN)
	{
	}
	point2d low;
	point2d high;
};

class convex_hull_approx_test
{
public:
	static constexpr int32_t NUM_POINTS = 50000;
	static constexpr double DISCRETIZATION_SIZE = 200;
	convex_hull_approx_test() = default;
	~convex_hull_approx_test() = default;

	/// Function to be optimized
	void optimize_this(const std::vector<point2d> &points, std::vector<point2d> &approx_convex_hull);

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

void convex_hull_approx_test::run_test()
{
	// Generate a bunch of uniformly random 2D points. 
	constexpr int32_t NUM_RUNS = 3000;
	std::vector<std::vector<point2d>> set_of_points;
	convex_hull_approx_test test;
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
		for (const auto &points : set_of_points)
		{
			std::vector<point2d> approx_convex_hull;
			test.optimize_this(points, approx_convex_hull);
		}
	}
}

/// Run approximation of convex_hull_test, adapted from http://geomalgorithms.com/a11-_hull-2.html 
void convex_hull_approx_test::optimize_this(const std::vector<point2d> &points, std::vector<point2d> &approx_convex_hull)
{
	// Find x-range of all points. 
	const auto minmax_x = std::minmax_element(std::begin(points), std::end(points),
		[](point2d const& p1, point2d const& p2)
	{
		return p1.x < p2.x;
	});

	// Histogram points by x-coordinate, saving the highest and lowest points (by y-coordinate) in each bin. 
	const double range_x = minmax_x.second->x - minmax_x.first->x;
	const int32_t num_bins = std::floor(range_x * DISCRETIZATION_SIZE) + 1;
	std::vector<point_bin> bins(num_bins);

	for (const auto &point : points)
	{
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
		if (bin.low.x == _DBL_MAX && bin.high.x == _DBL_MIN) {
			continue;
		}
		else if (bin.low.x == _DBL_MAX) {
			bin.low = bin.high;
		}
		else if (bin.low.x == _DBL_MIN) {
			bin.high = bin.low;
		}
		const auto &p_low = bin.low;
		

		while (index >= 2)
		{
			// http://stackoverflow.com/questions/1560492/how-to-tell-whether-a-point-is-to-the-right-or-left-side-of-a-line
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
		if (bin.low.x == _DBL_MAX && bin.high.x == _DBL_MIN) {
			continue;
		}
		else if (bin.low.x == _DBL_MAX) {
			bin.low = bin.high;
		}
		else if (bin.low.x == _DBL_MIN) {
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

int main()
{
	convex_hull_approx_test::run_test();
}
