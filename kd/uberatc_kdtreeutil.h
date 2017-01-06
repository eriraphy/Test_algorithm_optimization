// uberatc_kdtreeutil.h                                               -*-C++-*-
#ifndef INCLUDED_UBERATC_KDTREEUTIL
#define INCLUDED_UBERATC_KDTREEUTIL

#include "uberatc_point.h"

#include <assert.h>
#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

namespace uber {
namespace atc {

                              // ================
                              // class KDTreeUtil
                              // ================

class KDTreeUtil {
  public:
    // Compares two points 'pt1', and 'pt2' along dimension given by
    // 'dimension' and returns true if 'pt1' is less than 'pt2'.
    // The behavior of this method is undefined if 'pt1' and 'pt2' are points
    // in different dimensions, or if 'dimension' is greater than the
    // dimensions of the two points.
    template<typename VALUE>
    static bool compareInSingleDimension(
                               const std::shared_ptr<Point<VALUE> >& pt1,
                               const std::shared_ptr<Point<VALUE> >& pt2,
                               int                                   dimension)
    {
        return (*pt1)[dimension] < (*pt2)[dimension];
    }

    // Calculate the L2-norm distance between points 'pt1' and 'pt2', and
    // the squared distance between the two points.
    // The behavior of this method is undefined if 'pt1' and 'pt2' are points
    // in different dimensions.
    template<typename VALUE>
    static VALUE L2DistanceBetweenPoints(const Point<VALUE>& pt1,
                                         const Point<VALUE>& pt2)
    {
        VALUE dist(0);
        assert(pt1.size() == pt2.size());
        auto it1 = pt1.begin();
        auto it2 = pt2.begin();
        for (; it1 != pt1.end() && it2 != pt2.end();
            ++it1, ++it2) {
            VALUE diff = *it1 - *it2;
            dist += diff * diff;
        }
        return dist;
    }

    // Calculate the L2-norm distance between the point defined by 'point',
    // and the hyperplane along the axis defined by 'axis' at the location
    // 'value'. Returns the distance squared between the point and the
    // hyperplane. The behavior of this method is undefined if 'point' is in
    // a lower dimension than the hyperplane. 
    template<typename VALUE>
    static VALUE L2DistanceBetweenPointAndPlane(const Point<VALUE>& point,
                                                VALUE               value,
                                                int                 axis)
    {
        VALUE diff = point[axis] - value;
        return diff * diff;
    }

    // Defines a simple heuristic for deciding the axis and value along which
    // to split a KDTree node. Identify the axis along which the points defined
    // by 'points' have maximal variation, and pick the median value along the
    // axis to split. Returns a pair of the axis/dimension along which to split
    // and the median value. The behavior is undefined if the points in
    // 'points' are in different dimensions. It is also undefined behavior to
    // call this method with an empty list of points.
    template<typename VALUE>
    static typename std::pair<int, VALUE>
    SimpleSplittingHeuristic(
                          std::vector<std::shared_ptr<Point<VALUE> > >& points)
    {
        // Search for dimension with largest range
        int best_dimension = 0;
        VALUE max_diff = std::numeric_limits<VALUE>::min();

        int numDimensions = points[0]->size();
        for (int i = 0; i < numDimensions; ++i) {
            VALUE max_xi = std::numeric_limits<VALUE>::min();
            VALUE min_xi = std::numeric_limits<VALUE>::max();
            for (auto point : points) {
                VALUE p_i = (*point)[i];
                if (p_i < min_xi) {
                    min_xi = p_i;
                }
                if (p_i > max_xi) {
                    max_xi = p_i;
                }
            }

            if (max_xi - min_xi > max_diff) {
                best_dimension = i;
                max_diff = max_xi - min_xi;
            }
        }

        // Find the median of the values within that dimension
        using namespace std::placeholders;
        auto customCompare
            = std::bind(compareInSingleDimension<VALUE>,
                        _1,
                        _2,
                        best_dimension);

        sort(points.begin(), points.end(), customCompare);

        VALUE median_value;
        if (points.size() % 2 == 1) {
            // If there are an odd number of points, then there is a single
            // value which defines the median.
            auto median_it = points.begin() + (points.size() / 2);
            auto median_pt = *median_it;
            median_value = (*median_pt)[best_dimension];
        } else {
            // There are an even number of points so we take the average
            // of the middle two to be the median value.
            auto median_it = points.begin() + ((points.size() - 1) / 2);
            auto median_pt1 = *median_it;
            auto median_pt2 = *(++median_it);
            median_value = ((*median_pt1)[best_dimension]
                            + (*median_pt2)[best_dimension]) / 2;

        }
        return { best_dimension, median_value } ;
    }
};

} // close namespace atc
} // close namespace uber

#endif // INCLUDED_UBERATC_KDTREEUTIL
