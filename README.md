# Writing
- for panel, acknowledge that points may move (or be missing for some time periods)
    - just cache the neighbor pairs
- also provide the option to input projected data.
- once I have something working reasonably well in cython, contribute to `statsmodels` in python
- once I have something working reasonably well in cython, make an R package
- what if there are duplicate points?
- check if the haversine formula, when it's inaccurate, gives an answer large enough that it doesn't matter (and therefore it's unnecessary to use the full `atan2` formula)
    - if so, how much faster is it?
- Figure out how to parse missing data?

# Panel
- Warn about unbalanced data?
- Add caching for distance calcs

# Testing
- hypothesis

# Relevant websites
- https://en.wikipedia.org/wiki/Vincenty's_formulae
- https://en.wikipedia.org/wiki/Great-circle_distance
- http://statsmodels.sourceforge.net/stable/stats.html#sandwich-robust-covariances
- https://github.com/geopy/geopy/blob/master/geopy/distance.py
- https://github.com/thehackerwithin/berkeley/blob/b34ea54f7ff1dd9300ce7aa23cb78caf487a9a02/cython_spring16/setup_cy.py
- https://github.com/scikit-learn/scikit-learn/blob/06bf797c0deabe2a2f166d19abbd0c305da4d123/doc/developers/performance.rst
- http://opexanalytics.com/technical-note-on-calculating-many-distances-for-analytics-projects/


# Done
- if I don't import `geopy` directly, test against that
- use scikit-learn's license (BSD3)
- copy code from scikit-learn and get it to build
    - bring in the check_array function without the rest of the validate script
- Add a check that people are providing lat/long in decimal, not degrees, minutes, seconds
- Take a statsmodels expression and parse out the model, rather than using my own OLS implementation.
- Add a check that people provide a reasonable distance (non-negative and not covering the entire earth)
