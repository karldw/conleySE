# import sys, os
# myPath = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, myPath + '/../')
#
# from hypothesis import given
# from hypothesis.strategies import floats, tuples, integers
# import numpy as np
# from hypothesis.extra import numpy
# from geopy.distance import great_circle as geopy_great_circle
# from distance import great_circle as our_great_circle  # Local cython code
#
#
#
# hypoth_lat_long = tuples(floats(min_value = -90, max_value = 90),
#                          floats(min_value = -179.999999, max_value = 180))
#
# @given(hypoth_lat_long, hypoth_lat_long)
# def test_great_circle(pt1, pt2):
#     #np_formulation = np.array([pt1, pt2])
#     pt1_np = np.array(pt1)
#     pt2_np = np.array(pt2)
#     assert np.allclose(our_great_circle(pt1_np, pt2_np), geopy_great_circle(pt1, pt2).km)
#
#
# #
# # def rnd_len_arrays(dtype, min_len=0, max_len=3, elements=None):
# #     lengths = integers(min_value=min_len, max_value=max_len)
# #     return lengths.flatmap(lambda n: arrays(dtype, n, elements=elements))
# #
# #
# # @given(np.float, ())
# # def test_great_circle_arrays():
# #     pass
