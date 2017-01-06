// uberatc_point.h                                                    -*-C++-*-

#ifndef INCLUDED_UBERATC_POINT
#define INCLUDED_UBERATC_POINT

#include <vector>

namespace uber {
namespace atc {

                                // ===========
                                // class Point
                                // ===========

template <typename VALUE>
class Point {
    // This class provides a wrapper around a data point in space
  public:
    typedef typename std::vector<VALUE>::iterator       iterator;
    typedef typename std::vector<VALUE>::const_iterator const_iterator;

  private:
    const int          d_pointId;
        // A unique identifier for the data point

    std::vector<VALUE> d_data;
        // The actual data which represents the data point

  public:
    // CONSTRUCTORS
    Point(int pointId, std::vector<VALUE>& data);
        // Create a point with an id given by 'pointId' from the data
        // referenced by 'data'.

    // MANIPULATORS
    VALUE& operator[](std::size_t index);
    iterator begin();
    iterator end();

    // ACCESSORS
    int pointId() const;
        // Return the unique identifier associated with this point
    const std::vector<VALUE>& data() const;
    std::size_t size() const;
    const VALUE& operator[](std::size_t index) const;
    const_iterator begin() const;
    const_iterator end() const;
};

// ============================================================================
//                      INLINE FUNCTION DEFINITIONS
// ============================================================================



                            // -----------
                            // class Point
                            // -----------

template <typename VALUE>
Point<VALUE>::Point(int pointId, std::vector<VALUE>& data)
: d_pointId(pointId)
, d_data(data)
{
}

template<typename VALUE>
VALUE& Point<VALUE>::operator[](std::size_t index)
{
    return d_data[index];
}

template<typename VALUE>
typename Point<VALUE>::iterator Point<VALUE>::begin()
{
    return d_data.begin();
}

template<typename VALUE>
typename Point<VALUE>::iterator Point<VALUE>::end()
{
    return d_data.end();
}

template<typename VALUE>
int Point<VALUE>::pointId() const
{
    return d_pointId;
}

template<typename VALUE>
const std::vector<VALUE>& Point<VALUE>::data() const
{
    return d_data;
}

template<typename VALUE>
std::size_t Point<VALUE>::size() const
{
    return d_data.size();
}

template<typename VALUE>
const VALUE& Point<VALUE>::operator[](std::size_t index) const
{
    return d_data[index];
}

template<typename VALUE>
typename Point<VALUE>::const_iterator Point<VALUE>::begin() const
{
    return d_data.begin();
}

template<typename VALUE>
typename Point<VALUE>::const_iterator Point<VALUE>::end() const
{
    return d_data.end();
}



} // close namespace atc
} // close namespace uber

#endif // INCLUDED_UBERATC_POINT
