// uberatc_kdtree.h                                                   -*-C++-*-
#ifndef INCLUDED_UBERATC_KDTREE
#define INCLUDED_UBERATC_KDTREE

#include "uberatc_point.h"

#include <assert.h>
#include <algorithm>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept> // std::invalid_argument exception

namespace uber {
namespace atc {

static const int k_KDTreeVersionNum = 0x100;

                            // =================
                            // class KDTree_Node
                            // =================

template <typename VALUE>
class KDTree_Node {
    // Internal class representing a node for the KDTree

  public:
    typedef std::shared_ptr<KDTree_Node<VALUE> > NodePtr;
    typedef std::shared_ptr<Point<VALUE> >       PointPtr;

    struct NodeType {
        // A node can be of different types. A leaf node points to 'Point'
        // which describes a data point, whereas a non-leaf node is a node that
        // has two children, and contains a value which identifies the decision
        // boundry that node represents in the KDTree
        enum Type {
            e_none = 0,
                // An un-initialized node is neither
            e_leaf,
                // A leaf node contains data referenced by 'd_dataPoint'
            e_nonleaf
                // A non-leaf node has children and describes a decision
                // boundary.
        };
    };

    typename NodeType::Type d_nodeType;
        // The type of this node

    PointPtr d_dataPoint;
        // Pointer to the data referenced by this node, this is only valid if
        // this is a leaf node.

    NodePtr  d_leftNode;
        // The left child of this node
    NodePtr  d_rightNode;
        // The right child of this node
    int      d_splitDimension;
        // The dimension across which this node splits the space
    VALUE    d_splitValue;
        // Determines the value of the split

    // CONSTRUCTORS
    KDTree_Node();
        // Construct an empty node with un-initialized values and null
        // pointers.

    // MANIPULATORS
    void loadNode(std::istream& is);
        // Load the contents of this node, and it's subtree from the input
        // stream given by 'is'.

    // ACCESSORS
    void saveNode(std::ostream& os) const;
        // Output a binary representation of the subtree pointed to by this
        // node object to 'os'.
};


                              // ============
                              // class KDTree
                              // ============

template <typename VALUE>
class KDTree {
    // A KDTree of values defined by 'VALUE'. The KDTree is a binary tree which
    // provides container for storage of k-dimensional points such that
    // querying the tree for nearest neighbors, once initialized with data, is
    // efficient.

  public:
    typedef typename KDTree_Node<VALUE>::PointPtr  PointPtr;
    typedef typename KDTree_Node<VALUE>::NodePtr   NodePtr;
    typedef typename std::pair<int, VALUE>         SplitValue;

    typedef std::function<SplitValue(std::vector<PointPtr>&)>
                                                   SplitDecisionFunc;
      // Given a list of data points, determines the value and dimension on
      // which to split a level of the KDTree.

    typedef typename std::function<VALUE(const Point<VALUE>&,
                                         const Point<VALUE>&)>
                                                   DistanceFunc;
      // Given two points, returns the distance between those two points

    typedef typename std::function<VALUE(const Point<VALUE>&, VALUE, int)>
                                                   HyperplaneDistanceFunc;
      // Given a point, and a hyperplane defined by a value and the
      // dimension/axis, returns the distance between the hyperplane and
      // the point.

  private:

    struct BestNearestNeighborMatch {
        PointPtr d_point;
        VALUE    d_distance;
    };

    mutable std::mutex d_mutex;
        // Lock for this object
    bool               d_isInitialized;
        // True iff this instance has been initialized with data
    NodePtr            d_root;
        // Pointer to the root of this node
    int                d_numDimensions;
        // The number of dimensions of each point in this tree

    NodePtr initializeHelper(std::vector<PointPtr>&   dataset,
                             const SplitDecisionFunc& splitFunc);
        // Initialize a KDTree from the points given by 'dataset' using
        // 'splitFunc' to make the decision for splitting values at each
        // level of the tree, and return a pointer to the root of the tree.
        // The behavior is undefined if 'dataset' contains points of
        // different dimensions.

    void getNearestNeighborHelper(
                          const Point<VALUE>&           point,
                          const DistanceFunc&           distanceMeasure,
                          const HyperplaneDistanceFunc& distanceMeasureToPlane,
                          const NodePtr&                root,
                          BestNearestNeighborMatch&     bestMatchSoFar) const;
        // Search for the nearest neighbor of 'point' in the tree given by
        // 'root', using the 'distanceMeasure', and 'distanceToPlaneMeasure'
        // as measures of distance between two points, and a point and a
        // hyperplane respectively. If a point in the tree is closer than the
        // point given in 'bestMatchSoFar', update 'bestMatchSoFar' to
        // reference this point.

  public:
    // CREATORS
    KDTree();
        // Create an empty KDTree. Once constructed, the tree can be
        // initialized with data passed to 'initialize()', or by loading
        // contents from a file using 'loadFromFile()'.

    // MANIPULATORS
    int initialize(const std::vector<Point<VALUE> >& dataset,
                   const SplitDecisionFunc&          splitFunc);
        // Populate this KDTree with data points given by 'dataset'.
        // Returns 0 on success, and a non-zero value indicating failure if
        // this KDTree has already been initialized. It is undefined behavior
        // if 'dataset' contains points which are of varying dimensions.

    int loadFromFile(std::string& fileName);
        // Load a saved KDTree from the file referenced by 'fileName'.
        // Returns 0 on success, and a non-zero value indicating failure if
        // this KDTree has already been initialized.

    // ACCESSORS
    bool isInitialized() const;
        // Returns true if this tree has already been initialized with data

    int numDimensions() const;
        // Returns the number of dimensions of points within this KDTree if
        // the tree has been initialized, returns a negative value otherwise.

    const PointPtr
    getNearestNeighbor(
                   const Point<VALUE>&           point,
                   const DistanceFunc&           distanceMeasure,
                   const HyperplaneDistanceFunc& distanceMeasureToPlane) const;
        // Calculate the nearest neighbor of 'point' within the KDTree using
        // the distance measures defined by 'distanceMeasure' and
        // 'distanceMeasureToPlane'. Returns a pointer to the nearest
        // neighbor of 'point'. Returns a nullptr if the KDTree has not been
        // initialized. It is undefined behavior if 'point' isn't the same
        // dimensions as the points within the KDTree.

    int saveToFile(std::string& fileName) const;
        // Save a binary representation of the KDTree to the file referenced
        // by 'fileName'.
};

// ============================================================================
//                      INLINE FUNCTION DEFINITIONS
// ============================================================================


                        // -----------------
                        // class KDTree_Node
                        // -----------------

template <typename VALUE>
KDTree_Node<VALUE>::KDTree_Node()
: d_dataPoint(nullptr)
, d_leftNode(nullptr)
, d_rightNode(nullptr)
, d_splitDimension(-1)
{
    d_nodeType = NodeType::e_none;
}

template <typename VALUE>
void KDTree_Node<VALUE>::saveNode(std::ostream& os) const
{
    os.write(reinterpret_cast<const char*>(&d_nodeType),
             sizeof(d_nodeType));
    if (d_nodeType == NodeType::e_none) {
        return;
    } else if (d_nodeType == NodeType::e_leaf) {
        assert(d_dataPoint);
        int dataPtLen = d_dataPoint->size();
        int pointId = d_dataPoint->pointId();
        os.write(reinterpret_cast<const char*>(&pointId),
                 sizeof(pointId));
        os.write(reinterpret_cast<const char*>(&dataPtLen),
                 sizeof(dataPtLen));
        for (auto val : d_dataPoint->data()) {
            os.write(reinterpret_cast<const char*>(&val), sizeof(val));
        }
        return;

    }

    // Recursive case
    assert(d_leftNode);
    assert(d_rightNode);
    os.write(reinterpret_cast<const char*>(&d_splitDimension),
             sizeof(d_splitDimension));
    os.write(reinterpret_cast<const char*>(&d_splitValue),
             sizeof(d_splitValue));
    d_leftNode->saveNode(os);
    d_rightNode->saveNode(os);
}

template <typename VALUE>
void KDTree_Node<VALUE>::loadNode(std::istream& is)
{
    is.read(reinterpret_cast<char*>(&d_nodeType),
             sizeof(d_nodeType));
    if (d_nodeType == NodeType::e_none) {
        return;
    } else if (d_nodeType == NodeType::e_leaf) {
        int pointId, dataPtLen;
        is.read(reinterpret_cast<char*>(&pointId),
                sizeof(pointId));
        is.read(reinterpret_cast<char*>(&dataPtLen),
                sizeof(dataPtLen));
        std::vector<VALUE> data;
        data.reserve(dataPtLen);
        for (int i = 0; i < dataPtLen; ++i) {
            VALUE tmpVal;
            is.read(reinterpret_cast<char*>(&tmpVal), sizeof(tmpVal));
            data.push_back(tmpVal);
        }
        assert(data.size() == dataPtLen);
        d_dataPoint = PointPtr(new Point<VALUE>(pointId, data));
        return;
    }

    // recursive case
    is.read(reinterpret_cast<char*>(&d_splitDimension),
             sizeof(d_splitDimension));
    is.read(reinterpret_cast<char*>(&d_splitValue),
             sizeof(d_splitValue));
    d_leftNode = NodePtr(new KDTree_Node<VALUE>());
    d_rightNode = NodePtr(new KDTree_Node<VALUE>());
    d_leftNode->loadNode(is);
    d_rightNode->loadNode(is);
}

                            // ------------
                            // class KDTree
                            // ------------

template <typename VALUE>
KDTree<VALUE>::KDTree()
: d_isInitialized(false)
, d_root(nullptr)
, d_numDimensions(-1)
{
}

template <typename VALUE>
typename KDTree<VALUE>::NodePtr
KDTree<VALUE>::initializeHelper(std::vector<PointPtr>&   dataPoints,
                                const SplitDecisionFunc& splitFunc)
{
    NodePtr root = NodePtr(new KDTree_Node<VALUE>());
    if (dataPoints.size() == 0) {
        return root;
    }

    if (dataPoints.size() == 1) {
        // Base case
        root->d_nodeType = KDTree_Node<VALUE>::NodeType::e_leaf;
        root->d_dataPoint = dataPoints[0];
        return root;
    }

    SplitValue splitVal = splitFunc(dataPoints);
    int splitDimension = splitVal.first;
    VALUE splitValue = splitVal.second;
    if (splitDimension >= d_numDimensions) {
        throw std::runtime_error(
            "Axis to split on is out of range of the number of dimensions of" \
            " the KDTree");
    }

    std::vector<PointPtr> leftSubtreePts;
    std::vector<PointPtr> rightSubtreePts;
    for (auto& pointPtr : dataPoints) {
        const VALUE& pointValue
            = (*pointPtr)[splitDimension];
        if (pointValue <= splitValue) {
            leftSubtreePts.push_back(pointPtr);
        } else {
            rightSubtreePts.push_back(pointPtr);
        }
    }

    NodePtr leftSubtree = initializeHelper(leftSubtreePts, splitFunc);
    NodePtr rightSubtree = initializeHelper(rightSubtreePts, splitFunc);
    root->d_nodeType = KDTree_Node<VALUE>::NodeType::e_nonleaf;
    root->d_leftNode = leftSubtree;
    root->d_rightNode = rightSubtree;
    root->d_splitDimension = splitDimension;
    root->d_splitValue = splitValue;

    return root;


}

template <typename VALUE>
int
KDTree<VALUE>::initialize(const std::vector<Point<VALUE> >& dataset,
                          const SplitDecisionFunc&          splitFunc)
{
    if (dataset.size() == 0) {
        throw std::invalid_argument(
            "KDTree::initialize invoked with an empty dataset");
    }

    std::lock_guard<std::mutex> lockGuard(d_mutex);
    if (d_isInitialized || dataset.size() == 0) {
        return -1;
    }

    d_numDimensions = dataset[0].size();
    std::vector<PointPtr> dataPoints;
    for (auto& dataPt : dataset) {
        if (dataPt.size() != d_numDimensions) {
            return -1;
        }
        PointPtr point_sp = PointPtr(new Point<VALUE>(dataPt));
        dataPoints.push_back(point_sp);
    }

    d_root = initializeHelper(dataPoints, splitFunc);
    d_isInitialized = true;
    return 0;
}

template <typename VALUE>
int KDTree<VALUE>::loadFromFile(std::string& fileName)
{
    std::lock_guard<std::mutex> lockGuard(d_mutex);
    if (d_isInitialized) {
        return -1;
    }

    std::ifstream inFile(fileName, std::ifstream::binary);
    int checksum;
    inFile.read(reinterpret_cast<char*>(&checksum), sizeof(checksum));
    assert(checksum == k_KDTreeVersionNum);
    bool wasInitialized(false);
    inFile.read(reinterpret_cast<char*>(&wasInitialized),
                sizeof(wasInitialized));
    if (wasInitialized) {
        inFile.read(reinterpret_cast<char*>(&d_numDimensions),
                    sizeof(d_numDimensions));
        d_root = NodePtr(new KDTree_Node<VALUE>());
        d_root->loadNode(inFile);
        d_isInitialized = true;
    }
    inFile.close();
    return 0;
}

template <typename VALUE>
int KDTree<VALUE>::saveToFile(std::string& fileName) const 
{
    std::lock_guard<std::mutex> lockGuard(d_mutex);
    std::ofstream outFile(fileName, std::ofstream::binary);
    outFile.write(reinterpret_cast<const char*>(&k_KDTreeVersionNum),
                  sizeof(k_KDTreeVersionNum));
    outFile.write(reinterpret_cast<const char*>(&d_isInitialized),
                  sizeof(d_isInitialized));
    if (d_isInitialized) {
        outFile.write(reinterpret_cast<const char*>(&d_numDimensions),
                      sizeof(d_numDimensions));
        d_root->saveNode(outFile);
    }
    outFile.close();
    return 0;
}

template <typename VALUE>
bool KDTree<VALUE>::isInitialized() const
{
    std::lock_guard<std::mutex> lockGuard(d_mutex);
    return d_isInitialized;
}

template <typename VALUE>
int KDTree<VALUE>::numDimensions() const
{
    std::lock_guard<std::mutex> lockGuard(d_mutex);
    return d_numDimensions;
}


template<typename VALUE>
void KDTree<VALUE>::getNearestNeighborHelper(
                          const Point<VALUE>&           point,
                          const DistanceFunc&           distanceMeasure,
                          const HyperplaneDistanceFunc& distanceMeasureToPlane,
                          const NodePtr&                root,
                          BestNearestNeighborMatch&     bestMatchSoFar) const
{
    if (!root || root->d_nodeType == KDTree_Node<VALUE>::NodeType::e_none) {
        // If this is a null node, then we simply return
        return;
    }

    // Base case: root is a leaf node
    if (root->d_nodeType == KDTree_Node<VALUE>::NodeType::e_leaf) {
        VALUE distToRoot = distanceMeasure(point, *(root->d_dataPoint));
        if (distToRoot <= bestMatchSoFar.d_distance) {
            bestMatchSoFar.d_point = root->d_dataPoint;
            bestMatchSoFar.d_distance = distToRoot;
        }
        return;
    }

    assert(root->d_nodeType == KDTree_Node<VALUE>::NodeType::e_nonleaf);

    // Recursively search for closest node if we are not at a leaf node
    int splitDim = root->d_splitDimension;
    NodePtr nearSide;
    NodePtr farSide;
    if (point[splitDim] <= root->d_splitValue) {
        // Go down the left subtree first
        nearSide = root->d_leftNode;
        farSide = root->d_rightNode;
    } else {
        // Go down the right subtree first
        nearSide = root->d_rightNode;
        farSide = root->d_leftNode;
    }
    
    getNearestNeighborHelper(point,
                             distanceMeasure,
                             distanceMeasureToPlane,
                             nearSide,
                             bestMatchSoFar);

    // Check if the hypersphere around 'point' intersects the hyperplane
    // defined by the splitting condition of 'root'. If so, we have to go down
    // the other subtree to find the nearest neighbor.
    VALUE distToHyperPlane
        = distanceMeasureToPlane(point,
                                 root->d_splitValue,
                                 root->d_splitDimension);
    if (distToHyperPlane < bestMatchSoFar.d_distance) {
        getNearestNeighborHelper(point,
                                 distanceMeasure,
                                 distanceMeasureToPlane,
                                 farSide,
                                 bestMatchSoFar);
    }
}

template <typename VALUE>
const typename KDTree<VALUE>::PointPtr
KDTree<VALUE>::getNearestNeighbor(
                    const Point<VALUE>&           point,
                    const DistanceFunc&           distanceMeasure,
                    const HyperplaneDistanceFunc& distanceMeasureToPlane) const
{
	/*the lockguard here conflict to the Intel TBB*/
    /*std::lock_guard<std::mutex> lockGuard(d_mutex);*/

    if (!d_isInitialized) {
        return PointPtr(nullptr);
    }

    if (point.size() != d_numDimensions) {
        throw std::invalid_argument("The dimension of point does not match" \
                                    " the dimension of the KDTree");
    }

    VALUE maxDist = std::numeric_limits<VALUE>::max();
    BestNearestNeighborMatch bestMatchSoFar = { PointPtr(nullptr), maxDist };
    getNearestNeighborHelper(point,
                             distanceMeasure,
                             distanceMeasureToPlane,
                             d_root,
                             bestMatchSoFar);
    return bestMatchSoFar.d_point;
}

} // close namespace atc
} // close namespace uber

#endif // INCLUDED_UBERATC_KDTREE
