#ifndef COMMON_H_
#define COMMON_H_

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/boost.h>
#include <typeinfo>

// Directory Iterator
struct recursive_directory_range{
  typedef boost::filesystem::recursive_directory_iterator iterator;
  recursive_directory_range (boost::filesystem::path p) :p_ (p){}
  iterator begin (){return boost::filesystem::recursive_directory_iterator (p_);}
  iterator end () {return boost::filesystem::recursive_directory_iterator ();}
  boost::filesystem::path p_;
};

#endif
