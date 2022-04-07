# py_slam
RE510 - Python SLAM

# Pre-requisites

## install g2opy

```
$ git clone https://github.com/uoip/g2opy.git
$ cd g2opy
$ mkdir build
$ cd build
$ cmake ..
$ make -j8
$ cd ..
$ python setup.py install
```

## install scikit-learn

```
$ pip install scikit-learn
```


## (Optional) g2o_viewer

To visualize the result written in .g2o file format, you can use g2o_viewer

However, there is a visualization code implemented using plt.plot(). So, you don't need g2o_viewer.