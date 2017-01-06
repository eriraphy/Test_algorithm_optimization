#include "uberatc_point.h"
#include "uberatc_kdtree.h"
#include "uberatc_kdtreeutil.h"


#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
//#include "tbb/tbb.h"
#include <time.h>

#define NOMINMAX
#include <windows.h>


#define NUM_DIMS 3
#define NUM_PTS 10000
#define NUM_QUERIES 100000
#define RANDOM_NUM_SEED 1337

using namespace tbb;
using namespace uber::atc;
using namespace std;

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


template<typename T>
class querycomp {
	const vector<Point<T>> &query_pts;
	const shared_ptr<KDTree<T>> &tree_sp;
	
public:
	shared_ptr<Point<T>>* &nbrs;
	void operator() (const blocked_range<size_t>& r) const {
		for (auto i = r.begin(); i != r.end(); ++i)
		{
			auto &it = query_pts[i];
			nbrs[i] = tree_sp->getNearestNeighbor(it, KDTreeUtil::L2DistanceBetweenPoints<T>,
				KDTreeUtil::L2DistanceBetweenPointAndPlane<T>);
			
		}
	}
	querycomp(const vector<Point<T>> &q, const shared_ptr<KDTree<T>> &t, shared_ptr<Point<T>>* &n) :
		query_pts(q), tree_sp(t), nbrs(n) {}
};


template<typename T>
void query_tree(default_random_engine &generator, const shared_ptr<KDTree<T>> tree_sp, int num_queries)
{
	std::cout << "Querying tree" << std::endl;
	uniform_real_distribution<T> distribution(-1000., 1000.);
	vector<Point<T>> query_pts;
	query_pts.reserve(num_queries);
	const int num_dims = tree_sp->numDimensions();
	for (int i = 0; i < num_queries; ++i)
	{
		vector<T> point;
		for (int j = 0; j < num_dims; ++j)
		{

			point.push_back(distribution(generator));
		}
		query_pts.push_back(Point<T>(i, point));
	}

	shared_ptr<Point<T>>* nbrs = (shared_ptr<Point<T>>*)malloc(num_queries * sizeof(shared_ptr<Point<T>>));



	//const size_t gsize = 1000;

		cout << "Start calling get nearest neighbor" << endl;
		{
			measure_time_and_print measure_time("get nearest neighbor");
			for (int i = 0; i != 10; i++) {
				parallel_for(blocked_range<size_t>(0, query_pts.size()), querycomp<T>(query_pts, tree_sp, nbrs));
			}
		}
	


	//{
	//	measure_time_and_print measure_time("get nearest neighbor");
	//	for (auto it = query_pts.begin(); it != query_pts.end(); ++it)
	//	{
	//		for (int i = 0; i != 10; i++) {
	//			const auto nbr = tree_sp->getNearestNeighbor(*it, KDTreeUtil::L2DistanceBetweenPoints<T>, KDTreeUtil::L2DistanceBetweenPointAndPlane<T>);
	//		}
	//	}
	//}
		free(nbrs);
}

template<typename T>
void build_tree(default_random_engine &generator, shared_ptr<KDTree<T>> tree_sp, int num_dims, int num_pts)
{
    std::cout << "Initializing tree" << std::endl;
    vector<Point<T>> dataPts;
    dataPts.reserve(num_pts);
    uniform_real_distribution<T> distribution(-1000., 1000.);
    for (int i=0; i < num_pts; ++i)
    {
         vector<T> point;
         for (int j=0; j < num_dims; ++j)
         {
             auto random_num = distribution(generator);
             point.push_back(random_num);
         }
         dataPts.push_back(Point<T>(i, point));
    }
    
	cout << "Calling initialize" << endl;
	{
		measure_time_and_print measure_time("build nearest neighbor");
		tree_sp->initialize(dataPts, KDTreeUtil::SimpleSplittingHeuristic<T>);
	}
}

int main(int argc, char **argv)
{
    default_random_engine generator(RANDOM_NUM_SEED);
    auto tree_sp = make_shared<KDTree<double>>();
    // Build the KD-Tree with data
    build_tree<double>(generator, tree_sp, NUM_DIMS, NUM_PTS);
    // Query the KD-Tree...
    query_tree<double>(generator, tree_sp, NUM_QUERIES);
	

	//system("pause");
    return 0;
};
