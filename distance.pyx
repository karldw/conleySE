cimport numpy as np  # C-numpy
#from typedefs cimport double, ITYPE_t  #, DITYPE_t
from libc.math cimport sin, cos, atan2, sqrt

# I don't know what these mean!  -- KDW
#@cython.boundscheck(False)
#@cython.wraparound(False)

#cdef DTYPE_t PI = cnp.pi

cdef double EARTH_RADIUS = 6371.009

cdef inline double radians(double degrees):
    """Convert degrees to radians"""
    cdef double rad = degrees * 3.1415926 / 180.0
    return rad


#cdef inline double great_circle(double[:] pt1, double[:] pt2):
def great_circle(double[:] pt1, double[:] pt2):
    """Take two points on earth and return a distance.

    This function assumes a spherical earth, which is accurate to about 0.5%.
    Look at Vincenty distances if you want to do better.

    pt1 and p2 should be arrays of dimension (2,), ordered latitude then
    longitude, both measured in degrees in the range (-180, 180].
    The output, d, is in kilometers.
    """
    cdef double lat1 = radians(pt1[0])
    cdef double lng1 = radians(pt1[1])
    cdef double lat2 = radians(pt2[0])
    cdef double lng2 = radians(pt2[1])
    cdef double sin_lat1 = sin(lat1)
    cdef double cos_lat1 = cos(lat1)
    cdef double sin_lat2 = sin(lat2)
    cdef double cos_lat2 = cos(lat2)

    cdef double delta_lng = lng2 - lng1
    cdef double sin_delta_lng = sin(delta_lng)
    cdef double cos_delta_lng = cos(delta_lng)

    cdef double d = atan2(sqrt((cos_lat2 * sin_delta_lng) ** 2 +
                  (cos_lat1 * sin_lat2 -
                   sin_lat1 * cos_lat2 * cos_delta_lng) ** 2),
             sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_delta_lng)
    return EARTH_RADIUS * d
