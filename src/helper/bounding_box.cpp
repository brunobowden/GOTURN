// DO NOT COMMIT - hack to include CHECK definition
#include "caffe/caffe.hpp"
    
#include "bounding_box.h"

#include <cstdio>
#include "helper/image_proc.h"

// Uncomment line below if you want to use rectangles
#define VOT_RECTANGLE
#include "native/vot.h"

#include "helper.h"

// How much context to pad the image and target with (relative to the
// bounding box size).
const double kContextFactor = 2;

// Factor by which to scale the bounding box coordinates, based on the
// neural network default output range.
const double kScaleFactor = 10;

// Factor by which to scale the rotation per frame in BoundingBox::Draw
// to make small rotations more discernable.
// TODO: choose best value from experimentation
const double kDrawRotSpeedScale = 10;
const double kPI = 3.14159265;
const double kDegToRad = kPI / 180.0;
// Maximum range of rotation frame to frame, i.e. -180.0 .. +180.0
const double kRotationRange = 180.0;

// If true, the neural network will estimate the bounding box corners: (x1, y1, x2, y2)
// If false, the neural network will estimate the bounding box center location and size: (center_x, center_y, width, height)
const bool use_coordinates_output = true;

BoundingBox::BoundingBox() :
  scale_factor_(kScaleFactor),
  x1_(0.0),
  x2_(0.0),
  y1_(0.0),
  y2_(0.0),
  rot_speed_(0.0)
{
}

BoundingBox::BoundingBox(const VOTRegion& region, float rot_speed)
{
  // VOTRegion is given by left, top, width, and height.
  x1_ = region.get_x();
  y1_ = region.get_y();
  x2_ = region.get_x() + region.get_width();
  y2_ = region.get_y() + region.get_height();
  rot_speed_ = rot_speed;
}

void BoundingBox::GetRegion(VOTRegion* region) {
  // VOTRegion is given by left, top, width, and height.
  region->set_x(x1_);
  region->set_y(y1_);
  region->set_width(get_width());
  region->set_height(get_height());
}


BoundingBox::BoundingBox(const std::vector<float>& bounding_box)
  : scale_factor_(kScaleFactor)
{
  if (bounding_box.size() != 5) {
    printf("Error - bounding box vector has %zu elements\n",
           bounding_box.size());
    exit(-1);
  }

  if (use_coordinates_output) {
    // Set bounding box coordinates.
    x1_ = bounding_box[0];
    y1_ = bounding_box[1];
    x2_ = bounding_box[2];
    y2_ = bounding_box[3];
  } else {
    // Get bounding box in format: (center_x, center_y, width, height)
    const double center_x = bounding_box[0];
    const double center_y = bounding_box[1];
    const double width = bounding_box[2];
    const double height = bounding_box[3];

    // Convert (center_x, center_y, width, height) to (x1, y1, x2, y2).
    x1_ = center_x - width / 2;
    y1_ = center_y - height / 2;
    x2_ = center_x + width / 2;
    y2_ = center_y + height / 2;
  }
  rot_speed_ = bounding_box[4];
  // DO NOT COMMIT
  // ignore orientation for now
  // orientation_ = 0.0;
}

void BoundingBox::GetVector(std::vector<float>* bounding_box) const {
  if (use_coordinates_output) {
    // Convert bounding box into a vector format using (x1, y1, x2, y2).
    bounding_box->push_back(x1_);
    bounding_box->push_back(y1_);
    bounding_box->push_back(x2_);
    bounding_box->push_back(y2_);
  } else {
    // Convert bounding box into a vector format using (center_x, center_y, width, height).
    bounding_box->push_back(get_center_x());
    bounding_box->push_back(get_center_y());
    bounding_box->push_back(get_width());
    bounding_box->push_back(get_height());
  }
  bounding_box->push_back(rot_speed_);
  // DO NOT COMMIT
  // ignore orientation for now
  // bounding_box->push_back(orientation_);
}

void BoundingBox::Print() const {
  printf("Bounding box: x,y: %lf, %lf, %lf, %lf, w,h: %lf, %lf, rot_speed: %lf\n",
         x1_, y1_, x2_, y2_, get_width(), get_height(), rot_speed_);
}

void BoundingBox::Scale(const cv::Mat& image, BoundingBox* bbox_scaled) const {
  *bbox_scaled = *this;

  const int width = image.cols;
  const int height = image.rows;

  // Scale to: 0 to 1
  bbox_scaled->x1_ /= width;
  bbox_scaled->y1_ /= height;
  bbox_scaled->x2_ /= width;
  bbox_scaled->y2_ /= height;
  // -kRotationRange to +kRotationRange => 0 to 1
  bbox_scaled->rot_speed_ = 0.5 + (bbox_scaled->rot_speed_ / (2.0 * kRotationRange));
  // rotation
  // Important that rot_speed_ has similar variance as x & y. Otherwise
  // with combined bbox and rotation layers, the L1 norm will prioritize
  // training towards features with a larger variance. Need better
  // understanding of both distributions to balance this.

  // Scale to: 0 to scale_factor_
  bbox_scaled->x1_ *= scale_factor_;
  bbox_scaled->y1_ *= scale_factor_;
  bbox_scaled->x2_ *= scale_factor_;
  bbox_scaled->y2_ *= scale_factor_;
  bbox_scaled->rot_speed_ *= scale_factor_;
}

void BoundingBox::Unscale(const cv::Mat& image, BoundingBox* bbox_unscaled) const {
  *bbox_unscaled = *this;

  const int image_width = image.cols;
  const int image_height = image.rows;

  // Scale to: 0 to 1
  bbox_unscaled->x1_ /= scale_factor_;
  bbox_unscaled->y1_ /= scale_factor_;
  bbox_unscaled->x2_ /= scale_factor_;
  bbox_unscaled->y2_ /= scale_factor_;
  bbox_unscaled->rot_speed_ /= scale_factor_;

  // Scale to: original image coordinates
  bbox_unscaled->x1_ *= image_width;
  bbox_unscaled->y1_ *= image_height;
  bbox_unscaled->x2_ *= image_width;
  bbox_unscaled->y2_ *= image_height;
  // Scale to: -kRotationRange to +kRotationRange
  bbox_unscaled->rot_speed_ = (bbox_unscaled->rot_speed_ - 0.5) * 2.0 * kRotationRange;

  std::cout << "BB::Recenter  - Unscale before:  " << *this << "\n";
  std::cout << "BB::Recenter  - Unscale after:   " << *bbox_unscaled << "\n";
}

double BoundingBox::compute_output_width() const {
  // Get the bounding box width.
  const double bbox_width = (x2_ - x1_);

  // We pad the image by a factor of kContextFactor around the bounding box
  // to include some image context.
  const double output_width = kContextFactor * bbox_width;

  // Ensure that the output width is at least 1 pixel.
  return std::max(1.0, output_width);
}

double BoundingBox::compute_output_height() const {
  // Get the bounding box height.
  const double bbox_height = (y2_ - y1_);

  // We pad the image by a factor of kContextFactor around the bounding box
  // to include some image context.
  const double output_height = kContextFactor * bbox_height;

  // Ensure that the output height is at least 1 pixel.
  return std::max(1.0, output_height);
}

double BoundingBox::get_center_x() const {
  // Compute the bounding box center x-coordinate.
  return (x1_ + x2_) / 2;
}

double BoundingBox::get_center_y() const {
  // Compute the bounding box center y-coordinate.
  return (y1_ + y2_) / 2;
}

void BoundingBox::Recenter(const BoundingBox& search_location,
              const double edge_spacing_x, const double edge_spacing_y,
              BoundingBox* bbox_gt_recentered) const {

  // Location of bounding box relative to the focused image and edge_spacing.
  // NOTE: rotation should alter the x, y values. This needs to be fixed
  // before training will work with combined layers. Ok for now as only
  // rotation is being trained.
  bbox_gt_recentered->x1_ = x1_ - search_location.x1_ + edge_spacing_x;
  bbox_gt_recentered->y1_ = y1_ - search_location.y1_ + edge_spacing_y;
  bbox_gt_recentered->x2_ = x2_ - search_location.x1_ + edge_spacing_x;
  bbox_gt_recentered->y2_ = y2_ - search_location.y1_ + edge_spacing_y;
  // Rotation
  bbox_gt_recentered->rot_speed_ = rot_speed_ - search_location.rot_speed_;
/*
  std::cout << "BB::Recenter  - this:            " << *this << "\n";
  std::cout << "BB::Recenter  - search_location: " << search_location << "\n";
  std::cout << "BB::Recenter  - recentered:      " << *bbox_gt_recentered << "\n";
  std::cout << "BB::Recenter  - edge spacing:    " << edge_spacing_x << ", " << edge_spacing_y << "\n";
  //CHECK(0.2 > fabs(search_location.rot_speed_));
  //CHECK(0.2 > fabs(this->rot_speed_));
  //CHECK(0.2 > fabs(bbox_gt_recentered->rot_speed_));
*/
}

void BoundingBox::Uncenter(const cv::Mat& raw_image,
                           const BoundingBox& search_location,
                           const double edge_spacing_x, const double edge_spacing_y,
                           BoundingBox* bbox_uncentered) const {
  // Undo the effect of Recenter.
  // NOTE: rotation should alter the x, y values. This needs to be fixed
  // before training will work with combined layers. Ok for now as only
  // rotation is being trained.
  // TODO: Replace all the coordinate scalar math with matrix operations.
  // The complexity of Scale / Unscale, Uncenter / Recenter, Crop, Padding,
  // Tight / Non-Tight BBoxs and the like is both necessary and convoluted.
  // Replacing all the coordinate math with matrix operations should streamline
  // this. In part because rather than 4 lines for each individual coordinate,
  // it should be a single line for a matrix multiplication. Unclear if this
  // will present issues to some of the max / min logic.
  bbox_uncentered->x1_ = std::max(0.0, x1_ + search_location.x1_ - edge_spacing_x);
  bbox_uncentered->y1_ = std::max(0.0, y1_ + search_location.y1_ - edge_spacing_y);
  bbox_uncentered->x2_ = std::min(static_cast<double>(raw_image.cols), x2_ + search_location.x1_ - edge_spacing_x);
  bbox_uncentered->y2_ = std::min(static_cast<double>(raw_image.rows), y2_ + search_location.y1_ - edge_spacing_y);
  // Rotation
  bbox_uncentered->rot_speed_ = rot_speed_ + search_location.rot_speed_;

#if 1
  std::cout << "BB::Uncenter  - this:            " << *this << "\n";
  std::cout << "BB::Uncenter  - search_location: " << search_location << "\n";
  std::cout << "BB::Uncenter  - uncentered:      " << *bbox_uncentered << "\n";
  std::cout << "BB::Uncenter  - edge spacing:    " << edge_spacing_x << ", " << edge_spacing_y << "\n";
#endif
}

double BoundingBox::edge_spacing_x() const {
  const double output_width = compute_output_width();
  const double bbox_center_x = get_center_x();

  // Compute the amount that the output "sticks out" beyond the edge of the image (edge effects).
  // If there are no edge effects, we would have output_width / 2 < bbox_center_x, but if the crop is near the left
  // edge of the image then we would have output_width / 2 > bbox_center_x, with the difference
  // being the amount that the output "sticks out" beyond the edge of the image.
  return std::max(0.0, output_width / 2 - bbox_center_x);
}

double BoundingBox::edge_spacing_y() const {
  const double output_height = compute_output_height();
  const double bbox_center_y = get_center_y();

  // Compute the amount that the output "sticks out" beyond the edge of the image (edge effects).
  // If there are no edge effects, we would have output_height / 2 < bbox_center_y, but if the crop is near the bottom
  // edge of the image then we would have output_height / 2 > bbox_center_y, with the difference
  // being the amount that the output "sticks out" beyond the edge of the image.
  return std::max(0.0, output_height / 2 - bbox_center_y);
}

void BoundingBox::Draw(const int r, const int g, const int b, bool showRotation,
                       cv::Mat* image) const {
  // Get the top-left point.
  const cv::Point point1(x1_, y1_);

  // Get the bottom-rigth point.
  const cv::Point point2(x2_, y2_);

  // Get the selected color.
  const cv::Scalar box_color(b, g, r);

  // Draw a rectangle corresponding to this bbox with the given color.
  const int thickness = 3;
  cv::rectangle(*image, point1, point2, box_color, thickness);

  if (showRotation) {
    // Rotation Speed
    // Draw a partial circle around center
    // start angle: vertical, end angle: proportional to rotation
    // DO NOT COMMIT - use blue for debugging, switch later to bbox color
    const cv::Scalar rot_color(255, 0, 0);
    const float maxwh = std::max(get_width(), get_height());
    const cv::Size axes(maxwh * 0.4, maxwh * 0.4);
    const cv::Point center(get_center_x(), get_center_y());
    const double end_angle = rot_speed_ * kDegToRad;
    const double rot_speed_scaled = rot_speed_ * kDrawRotSpeedScale;

    double startAngle = -90.0;
    double endAngle = -90.0 + rot_speed_scaled;
    cv::ellipse(*image, center, axes, 0.0, startAngle, endAngle, rot_color, thickness);
    // http://docs.opencv.org/3.0-beta/modules/imgproc/doc/drawing_functions.html#ellipse
 
    const cv::Point rot_line_end(
        center.x + (axes.height * sin(rot_speed_scaled * kDegToRad)),
        center.y - (axes.height * cos(rot_speed_scaled * kDegToRad)));
    cv::line(*image, center, rot_line_end, rot_color, thickness);
      
    // std::cout << "BB::Draw      - this:            " << *this << "\n";
    // CHECK(false);
    // Orientation
    // Draw a line from center to 1.5 maxwh above center
    // Rotate clockwise around bbox center
    /* untrained for now
    const double orientation_radians = orientation_ * kDegToRad;
    const cv::Point orientation_line_end(
        get_center_x() + (maxwh * 1.0 * sin(orientation_radians)),
        get_center_y() - (maxwh * 1.0 * cos(orientation_radians)));
    cv::line(*image, center, orientation_line_end, box_color, thickness);
    */
  }
}

void BoundingBox::DrawBoundingBox(cv::Mat* image) const {
  // Draw a white bounding box on the image.
  Draw(255, 255, 255, false, image);
}

void BoundingBox::Shift(const cv::Mat& image,
                        const double lambda_scale_frac,
                        const double lambda_shift_frac,
                        const double min_scale, const double max_scale,
                        const double lambda_rotation_frac,
                        const bool shift_motion_model,
                        BoundingBox* bbox_rand) const {
  const double width = get_width();
  const double height = get_height();

  double center_x = get_center_x();
  double center_y = get_center_y();

  // Number of times to try shifting the bounding box.
  const int kMaxNumTries = 10;

  // Sample a width scaling factor for the new crop window, thresholding the scale to stay within a reasonable window.
  double new_width = -1;
  int num_tries_width = 0;
  while ((new_width < 0 || new_width > image.cols - 1) && num_tries_width < kMaxNumTries) {
    // Sample.
    double width_scale_factor;
    if (shift_motion_model) {
      width_scale_factor = max(min_scale, min(max_scale, sample_exp_two_sided(lambda_scale_frac)));
    } else {
      const double rand_num = sample_rand_uniform();
      width_scale_factor = rand_num * (max_scale - min_scale) + min_scale;
    }
    // DO NOT COMMIT
    //printf("lambda_scale_frac: %lf, (min: %lf, max: %lf), wsf: %lf, sample: %lf\n",
    //       lambda_scale_frac, min_scale, max_scale, width_scale_factor, 
    //       sample_exp_two_sided(lambda_scale_frac));
    // Expand width by scaling factor.
    new_width = width * (1 + width_scale_factor);
    // Ensure that width stays within valid limits.
    new_width = max(1.0, min(static_cast<double>(image.cols - 1), new_width));
    num_tries_width++;
  }

  // Find a height scaling factor for the new crop window, thresholding the scale to stay within a reasonable window.
  double new_height = -1;
  int num_tries_height = 0;
  while ((new_height < 0 || new_height > image.rows - 1) && num_tries_height < kMaxNumTries) {
    // Sample.
    double height_scale_factor;
    if (shift_motion_model) {
      height_scale_factor = max(min_scale, min(max_scale, sample_exp_two_sided(lambda_scale_frac)));
    } else {
      const double rand_num = sample_rand_uniform();
      height_scale_factor = rand_num * (max_scale - min_scale) + min_scale;
    }
    // Expand height by scaling factor.
    new_height = height * (1 + height_scale_factor);
    // Ensure that height stays within valid limits.
    new_height = max(1.0, min(static_cast<double>(image.rows - 1), new_height));
    num_tries_height++;
  }

  // Rotation
  // lambda_rotation_frac=12 => 1/12 of kRotationRange => +/- 15 degree average
  // clamped to -kRotationRange to +kRotationRange range
  double rotation = kRotationRange * max(-1.0, min(1.0, sample_exp_two_sided(lambda_rotation_frac)));
  // DO NOT COMMIT
  // See line 51 onwards in scripts/train.sh for a discussion on how to
  // model the rotation distribution. Instead of 5% large rotations, use
  // 50% to get faster training speed for large rotations during debugging
  // Probably should be command line variable like LAMBDA_ROTATION
  if (sample_rand_uniform() < 0.5) {
    rotation /= 10.0;
  }

  // DO NOT COMMIT
  //printf("lambda_rotation_frac: %lf, rotation: %lf\n",
  //         lambda_rotation_frac, rotation);

  // Ensure that new bounding box is within image even with rotation
  // rotation == 0.0 or rotation == 180.0:
  //    padding_x = new_width / 2
  //    padding_y = new_height / 2
  // rotation == 90.0:  (width and height are swapped)
  //    padding_x = new_height / 2
  //    padding_y = new_width / 2
  // The two 'fabs' are due to picking the outlier of 4 corners
  // Rotation causes distortion of the bounding box which is ignored.
  // For small rotations, e.g. 15 degrees, assumption is made that
  // retaining the same bounding box is reasonably approximation.
  // That also works for now as the bounding box weights are fixed and
  // only the rotation weights are being trained.
  const double padding_x =  // should it be int?
      fabs((new_width / 2) * cos(rotation * kDegToRad)) +
      fabs((new_height / 2) * sin(rotation * kDegToRad));
  const double padding_y =
      fabs((new_width / 2) * sin(rotation * kDegToRad)) +
      fabs((new_height / 2) * cos(rotation * kDegToRad));
  // DO NOT COMMIT
  //printf("rotation: %lf, new_width: %lf, new_height: %lf, padding_x: %lf, padding_y: %lf\n",
  //       rotation, new_width, new_height, padding_x, padding_y);

  // Find a random x translation for the new crop window.
  bool first_time_x = true;
  double new_center_x = -1;
  int num_tries_x = 0;
  while ((first_time_x ||
         // Ensure that the new object center remains in the old image window.
         new_center_x < center_x - width * kContextFactor / 2 ||
         new_center_x > center_x + width * kContextFactor / 2 ||
          // Ensure that the new window stays within the borders of the image.
         new_center_x - padding_x < 0 ||
         new_center_x + padding_x > image.cols)
         && num_tries_x < kMaxNumTries) {
    // Sample.
    double new_x_temp;
    if (shift_motion_model) {
      new_x_temp = center_x + width * sample_exp_two_sided(lambda_shift_frac);
    } else {
      const double rand_num = sample_rand_uniform();
      new_x_temp = center_x + rand_num * (2 * new_width) - new_width;
    }

    // Make sure that the window stays within the image.
    new_center_x = min(image.cols - padding_x, max(padding_x, new_x_temp));
    first_time_x = false;
    num_tries_x++;
    // DO NOT COMMIT
    //printf("lambda_shift_frac: %lf, width: %lf, center_x: %lf, new_center_x: %lf\n",
    //       lambda_shift_frac, width, center_x, new_center_x);
  }

  // Find a random y translation for the new crop window.
  bool first_time_y = true;
  double new_center_y = -1;
  int num_tries_y = 0;
  while ((first_time_y ||
          // Ensure that the new object center remains in the old image window.
         new_center_y < center_y - height * kContextFactor / 2 ||
         new_center_y > center_y + height * kContextFactor / 2  ||
          // Ensure that the new window stays within the borders of the image.
         new_center_y - padding_y < 0 ||
         new_center_y + padding_y > image.rows)
         && num_tries_y < kMaxNumTries) {
    // Sample.
    double new_y_temp;
    if (shift_motion_model) {
      new_y_temp = center_y + height * sample_exp_two_sided(lambda_shift_frac);
    } else {
      const double rand_num = sample_rand_uniform();
      new_y_temp = center_y + rand_num * (2 * new_height) - new_height;
    }
    // Make sure that the window stays within the image.
    new_center_y = min(image.rows - padding_y, max(padding_y, new_y_temp));
    first_time_y = false;
    num_tries_y++;
  }

  // Create a bounding box that matches the new sampled window.
  bbox_rand->x1_ = new_center_x - new_width / 2;
  bbox_rand->x2_ = new_center_x + new_width / 2;
  bbox_rand->y1_ = new_center_y - new_height / 2;
  bbox_rand->y2_ = new_center_y + new_height / 2;
  // rotation
  // object rotation, not bbox rotation
  bbox_rand->rot_speed_ = rotation;
  // Disable training for this by setting value to zero.
  // Need to generate more extreme rotations, e.g. object upside down to
  // effectively train orientation. Only rot_speed is being trained for now.
  // bbox_rand->orientation_ = 0.0;
}

std::ostream & operator<<(std::ostream &os, const BoundingBox& bbox) {
  // Aligns decimal points for ease of comparison across multiple lines
  // bbox:  24.5,  24.5,  73.9,  73.4, rot_speed:  2.6
  // bbox: 163.4,  93.5, 262.3, 191.4, rot_speed: -1.4
  // Misalignment ok for >1000.0 to allow better format for <1000 (common case)
  // bbox: 1634.2,  93.5, 262.2, 191.3, rot_speed: -2.3
  const int width = 5;  // 3 digits + decimal point + precision 1
  return os << std::fixed << std::setprecision(1) << "bbox: "
      << std::setw(width) << std::right << bbox.x1_ << ", "
      << std::setw(width) << std::right << bbox.y1_ << ", "
      << std::setw(width) << std::right << bbox.x2_ << ", "
      << std::setw(width) << std::right << bbox.y2_
      << ", rot_speed: " << std::setprecision(2) << std::setw(width)
      << std::right << bbox.rot_speed_;
}

double BoundingBox::compute_intersection(const BoundingBox& bbox) const {
  const double area = std::max(0.0, std::min(x2_, bbox.x2_) - std::max(x1_, bbox.x1_)) * std::max(0.0, std::min(y2_, bbox.y2_) - std::max(y1_, bbox.y1_));
  return area;
}

double BoundingBox::compute_area() const {
  return get_width() * get_height();
}
