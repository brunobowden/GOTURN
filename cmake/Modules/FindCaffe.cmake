# Caffe package
unset(Caffe_FOUND)

### Set the variable Caffe_DIR as the root of your caffe directory
# set(Caffe_DIR /path_to_caffe/build/install)
# DO NOT COMMIT
set(Caffe_DIR /home/ubuntu/caffe-gpu/distribute)

find_path(Caffe_INCLUDE_DIRS NAMES caffe/caffe.hpp caffe/common.hpp caffe/net.hpp caffe/proto/caffe.pb.h caffe/util/io.hpp caffe/vision_layers.hpp
  HINTS
  ${Caffe_DIR}/include)
set(Caffe_INCLUDE_DIRS /home/ubuntu/caffe-gpu/distribute/include)
message("Caffe_INCLUDE_DIRS: ${Caffe_INCLUDE_DIRS}")


find_library(Caffe_LIBRARIES NAMES caffe
  HINTS
  ${Caffe_DIR}/build/lib)
# DO NOT COMMIT
# Find the /usr/local/lib/libcaffe.so version
# set(Caffe_LIBRARIES /home/ubuntu/caffe-gpu/distribute/lib/libcaffe.so)
set(Caffe_LIBRARIES /home/ubuntu/pynb/caffe-gpu/.build_release/lib/libcaffe.so)

message("Hint: ${Caffe_DIR}/build/lib")
message("lib_dirs:${Caffe_LIBRARIES}")

if(Caffe_LIBRARIES AND Caffe_INCLUDE_DIRS)
    set(Caffe_FOUND 1)
endif()
