[![Build Status](https://travis-ci.org/eric7237cire/poker.svg?branch=master)](https://travis-ci.org/eric7237cire/poker)

This repo contains code to parse poker game information from a screenshot.

It's a sandbox for image processing / manipulation using the nice options available to python, like

1.  [Shapely](https://toblerity.org/shapely/manual.html)
2.  [Pillow](https://python-pillow.org/)
3.  [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/)
4.  [Numpy](http://www.numpy.org/)



Looking into poker evaluation libaries:

1.  [SKPokerEval.  A fast and lightweight 32-bit Texas Hold'em 7-card hand evaluator written in C++.](https://github.com/kennethshackleton/SKPokerEval)
2.  [Algo explanation](http://suffe.cool/poker/evaluator.html)
3.  [XPokerEval](https://web.archive.org/web/20111103160502/http://www.codingthewheel.com/archives/poker-hand-evaluator-roundup#2p2)
4.  [PokerStove](https://github.com/andrewprock/pokerstove)


Features:

1.  Features lots of contour processing to detect digits and cards
2.  Captures screenshots in c++ code, passed to python as a numpy RGB array
3.  Integrated 2+2 Poker evaluation from XPokerEval using a Monte Carlo simulation in a C++ compiled python extension

Built/run in windows using:

1)  Anaconda (included the .h files needed for numpy and python extending)
2)  Visual Studio Community 2017

Tests are run on travis with a linux environment


