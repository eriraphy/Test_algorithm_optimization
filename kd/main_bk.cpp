#include "uberatc_point.h"
#include "uberatc_kdtree.h"
#include "uberatc_kdtreeutil.h"

#include <chrono>
#include <iostream>
#include <limits>
#include <memory>
#include <random>

#define NUM_DIMS 3
#define NUM_PTS 10000
#define NUM_QUERIES 100000
#define RANDOM_NUM_SEED 1337

using namespace uber::atc;
using namespace std;

template<typename T>
void query_tree(default_random_engine &generator, const shared_ptr<KDTree<T>> tree_sp, int num_queries)
{
    std::cout << "Querying tree" << std::endl;
    uniform_real_distribution<T> distribution(-1000., 1000.);
    vector<Point<T>> query_pts;
    query_pts.reserve(num_queries);
    const int num_dims = tree_sp->numDimensions();
    for (int i=0; i < num_queries; ++i)
    {
         vector<T> point;
         for (int j=0; j < num_dims; ++j)
         {

             point.push_back(distribution(generator));
         }
         query_pts.push_back(Point<T>(i, point));
    }
    cout << "Start calling get nearest neighbor" << endl;
    auto begin_time = chrono::high_resolution_clock::now();
    for (auto it = query_pts.begin(); it != query_pts.end(); ++it)
    {
        // Optimize this call!
        const auto nbr = tree_sp->getNearestNeighbor(*it, KDTreeUtil::L2DistanceBetweenPoints<T>, KDTreeUtil::L2DistanceBetweenPointAndPlane<T>); 
    }
    auto end_time = chrono::high_resolution_clock::now();
    cout << "Completed query of " << num_queries << " points in: " << chrono::duration_cast<chrono::milliseconds>(end_time-begin_time).count() << " ms" << endl;
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
    auto begin_time = chrono::high_resolution_clock::now();
    tree_sp->initialize(dataPts, KDTreeUtil::SimpleSplittingHeuristic<T>);
    auto end_time = chrono::high_resolution_clock::now();
    cout << "Completed building of " << num_pts << " points in: " << chrono::duration_cast<chrono::milliseconds>(end_time-begin_time).count() << " ms" << endl;
}

int main(int argc, char **argv)
{
    default_random_engine generator(RANDOM_NUM_SEED);
    auto tree_sp = make_shared<KDTree<double>>();
    // Build the KD-Tree with data
    build_tree<double>(generator, tree_sp, NUM_DIMS, NUM_PTS);
    // Query the KD-Tree...
    query_tree<double>(generator, tree_sp, NUM_QUERIES);
    return 0;
};
