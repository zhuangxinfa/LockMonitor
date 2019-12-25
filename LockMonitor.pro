#-------------------------------------------------
#
# Project created by QtCreator 2019-10-10T09:38:59
#
#-------------------------------------------------

QT       += core gui printsupport network


greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = LockMonitor
TEMPLATE = app
# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS \
_GLIBCXX_USE_CXX11_ABI=0

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0


SOURCES += \
        main.cpp \
        lockmonitor.cpp \
    qcustomplot.cpp \
    lockmonitor2.cpp

HEADERS += \
        lockmonitor.h \
    lockmonitor.h \
    qcustomplot.h

FORMS += \
        lockmonitor.ui

INCLUDEPATH += /usr/include/opencv \
            /usr/include/hdf5/serial \
            /usr/local/include \
             /usr/local/cuda/include \
             /home/nvidia/caffe/include  \
             /usr/include/opencv \
            /usr/include/hdf5/serial \
            /usr/local/include \
            /usr/include/eigen3  \
            /home/nvidia/tensorflow    \
            /home/nvidia/tensorflow/bazel-genfiles \
            /home/nvidia/tensorflow/tensorflow  \
            /home/nvidia/tensorflow/third-party \
            /home/nvidia/tensorflow/tensorflow/contrib/makefile/downloads/absl  \
            /home/nvidia/tensorflow/tensorflow/contrib/makefile/downloads/protobuf/src

LIBS += /usr/lib/libopencv_video.so  \
  /home/nvidia/tensorflow/bazel-bin/tensorflow/libtensorflow_cc.so \
  /home/nvidia/tensorflow/bazel-bin/tensorflow/libtensorflow_framework.so \
  /usr/lib/libopencv_objdetect.so \
  /usr/lib/libopencv_ml.so  \
  /usr/lib/libopencv_core.so \
  /usr/lib/libopencv_features2d.so  \
  /usr/lib/libopencv_imgproc.so \
  /usr/lib/libopencv_highgui.so \
  /usr/lib/libopencv_flann.so   \
  /usr/lib/libopencv_imgcodecs.so  \
  /usr/lib/libopencv_calib3d.so \
    /usr/lib/libopencv_shape.so \
    /usr/lib/libopencv_videoio.so \
    /usr/lib/libopencv_core.so.3.3 \
  /usr/lib/libopencv_video.so  \
  /usr/lib/libopencv_objdetect.so \
  /usr/lib/libopencv_ml.so  \
  /usr/lib/libopencv_core.so \
  /usr/lib/libopencv_features2d.so  \
  /usr/lib/libopencv_imgproc.so \
  /usr/lib/libopencv_highgui.so \
  /usr/lib/libopencv_flann.so   \
  /usr/lib/libopencv_imgcodecs.so  \
  /usr/lib/libopencv_calib3d.so \



