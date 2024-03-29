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
- Consider making the residuals all at once: e = I-(X(X'X) ^(-1)X'y
- It's probably overkill to use the heavy-duty haversine function; I think there's no difference for points within the same quadrant of the globe. (Test this.)
- Consider using (pysparse's skyline format)[http://pysparse.sourceforge.net/formats.html#sparse-skyline-format] to work with symmetric sparse matrices
- Consider just importing the ball tree from scikit-learn, then scaling the unit sphere (with haversine distance) to the earth. (or, actually, downsizing the search radius to fit a unit sphere). The downside there is I couldn't support Vincenty distance.

# Panel
- Warn about unbalanced data?
- Add caching for distance calcs
- Use improved NW errors? https://www.federalreserve.gov/pubs/ifdp/2012/1060/ifdp1060.pdf
- Look at different dummy codings: http://statsmodels.sourceforge.net/devel/contrasts.html?highlight=fixed%20effects
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
- https://stackoverflow.com/questions/34189761/distribute-pre-compiled-cython-code-to-windows

# Done
- if I don't import `geopy` directly, test against that
- use scikit-learn's license (BSD3)
- copy code from scikit-learn and get it to build
    - bring in the check_array function without the rest of the validate script
- Add a check that people are providing lat/long in decimal, not degrees, minutes, seconds
- Take a statsmodels expression and parse out the model, rather than using my own OLS implementation.
- Add a check that people provide a reasonable distance (non-negative and not covering the entire earth)
