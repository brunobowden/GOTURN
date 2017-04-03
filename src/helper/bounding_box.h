#ifndef BOUNDING_BOX_H
#define BOUNDING_BOX_H

#include <vector>

#include <opencv/cv.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

class VOTRegion;

// Represents a bounding box on an image, with some additional functionality.
class BoundingBox
{
public:
  BoundingBox();
  // Encountered a bug due to trying "std::cout <<" for a "vector<float>".
  // Compiler used implicit conversion of vector<float> to a BoundingBox to use
  // operator<< function for BoundingBox. This printed a BoundingBox when it
  // was only a vector<float>. The explicit keyword prevents this.
  // Explicit should generally be used on all single argument constructors as a
  // precautionary measure to avoid bugs from unintended implicit conversions.
  // http://stackoverflow.com/questions/121162/what-does-the-explicit-keyword-mean-in-c/121163#121163
  explicit BoundingBox(const std::vector<float>& bounding_box);
  BoundingBox(const VOTRegion& region, float rot = 0.0);

  // Convert bounding box into a vector format.
  void GetVector(std::vector<float>* bounding_box) const;

  // Convert bounding box into VOTRegion format.
  void GetRegion(VOTRegion* region);

  // Print the bounding box coordinates.
  void Print() const;

  // Draw a rectangle corresponding to this bbox with the given color.
  void Draw(const int r, const int g, const int b, bool showRotation, cv::Mat* image) const;

  // Draw a white rectangle corresponding to this bbox.
  void DrawBoundingBox(cv::Mat* figure_ptr) const;

  // Normalize the size of the bounding box based on the size of the image.
  void Scale(const cv::Mat& image, BoundingBox* bbox_scaled) const;

  // Unnormalize the size of the bounding box based on the size of the image.
  // (Undoes the effect of Scale).
  void Unscale(const cv::Mat& image, BoundingBox* bbox_unscaled) const;

  // Compute location of bounding box relative to search region
  // edge_spacing_x and edge_spacing_y is the spacing of the image within the search region to account for edge effects.
  // *this should be the ground-truth bbox.
  void Recenter(const BoundingBox& search_location,
                const double edge_spacing_x, const double edge_spacing_y,
                BoundingBox* bbox_recentered) const;

  // Undo the effect of Recenter.
  void Uncenter(const cv::Mat& raw_image, const BoundingBox& search_location,
                const double edge_spacing_x, const double edge_spacing_y,
                BoundingBox* bbox_uncentered) const;

  // Shift the cropped region of the image to generate a new random training example.
  void Shift(const cv::Mat& image,
             const double lambda_scale_frac, const double lambda_shift_frac,
             // Unclear on what the "_frac" suffix means, "fractional" perhaps...
             // appears to be the same value as lambda_scale, lambda_shift and so on
             const double min_scale, const double max_scale, const double lambda_rotation_frac,
             const bool shift_motion_model,
             BoundingBox* bbox_rand) const;

  double get_scale_factor() const { return scale_factor_; }
  double get_width() const { return x2_ - x1_;  }
  double get_height() const { return y2_ - y1_; }

  // Compute the bounding box center x and y coordinates.
  double get_center_x() const;
  double get_center_y() const;

  // Get the size of the output image, which is equal to the bounding box with some padding.
  double compute_output_height() const;
  double compute_output_width() const;

  // Get the amount that the output "sticks out" beyond the left and bottom edges of the image.
  // This might be 0, but it might be > 0 if the output is near the edge of the image.
  double edge_spacing_x() const;
  double edge_spacing_y() const;

  // Area enclosed by the bounding box.
  double compute_area() const;

  // Area of intersection between two bounding boxes.
  double compute_intersection(const BoundingBox& bbox) const;

  // Bounding box coordiantes: top left, bottom right.
  double x1_, y1_, x2_, y2_;

  // Clockwise object rotation speed in degrees per frame, not rotation
  // of bbox itself. Likely more robust and accurate than orientation_
  // but greater error if integrated over time.
  // TODO: switch to OpenCV convention of ANTI-clockwise degrees rotation
  // http://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#getrotationmatrix2d
  // Could use name angular_velocity_ but that implies radians
  double rot_speed_;

  // DO NOT COMMIT
  // Ignore for now but may experiment with later
  // Clockwise orientation of object within bbox, not bbox orientation
  //   0.0 => human standing upright
  //  90.0 => human lying down, head to right
  // 180.0 => human handstand
  // This feature is likely to have a higher error than rot_speed_, especially
  // ambiguous case like the ball video, where there is no clear "upright"
  // but lower cumulative error if integrated over time in non-ambiguous cases.
  // double orientation_;

  // Factor to scale the bounding box coordinates before inputting into the neural net.
  double scale_factor_;
};

// std::cout << bbox_object;
std::ostream& operator<<(std::ostream &os, const BoundingBox& bbox);

#endif // BOUNDING_BOX_H
