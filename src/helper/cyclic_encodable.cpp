#include "cyclic_encodable.h"

#include <cmath>
#include <iomanip>

#include <glog/logging.h>

    
// Smaller ranges are possible but more likely a coding error
const double kSuspiciouslySmallRange = 0.9;

// TODO: use delegated constructors with C++ 11
// https://en.wikipedia.org/wiki/C++11#Object_construction_improvement
CyclicEncodable::CyclicEncodable(const double value, const double rangeMax):
  rangeMin_(0.0),
  rangeMax_(rangeMax)
{
  CHECK_LT(rangeMin_, rangeMax_);  // More descriptive error vs. CHECK_GE below
  CHECK_GT(getRange(), kSuspiciouslySmallRange);
  setValue(value);
}

CyclicEncodable::CyclicEncodable(
    const double value, const double rangeMin, const double rangeMax):
  rangeMin_(rangeMin),
  rangeMax_(rangeMax)
{
  CHECK_LT(rangeMin_, rangeMax_);
  CHECK_GT(getRange(), kSuspiciouslySmallRange);
  setValue(value);
}

CyclicEncodable::CyclicEncodable(const CyclicEncodable& other):
  rangeMin_(other.rangeMin_),
  rangeMax_(other.rangeMax_),
  theta_(other.theta_)
{
}

// Assignment operator can't change const values (rangeMin_ and rangeMax_).
// Instead it checks for the same range, the benefit is preventing copying
// between CyclicEncodable objects with different ranges, which is likely
// a programmer error. E.g. angle_object = time_object. These "type check"
// could be enforced by using a template but that's overkill for this issue.
CyclicEncodable& CyclicEncodable::operator=(const CyclicEncodable& other)
{
  // self-assignment check
  if (&other == this)
    return *this;
  // Expect exact match as ranges aren't modified like theta
  CHECK_EQ(rangeMin_, other.rangeMin_)
      << "CyclicEncodable assignment to object with different rangeMin";
  CHECK_EQ(rangeMax_, other.rangeMax_)
      << "CyclicEncodable assignment to object with different rangeMax";
  theta_ = other.theta_;
  return *this;
}

void CyclicEncodable::setValue(const double value)
{
  // Could be removed but likely useful to keep
  CHECK_LE(value, rangeMax_);
  CHECK_GE(value, rangeMin_);

  // Redundant with above CHECKs but kept as a precaution
  double wrapped = fmod(value - rangeMin_, getRange());
  double normalized = wrapped / getRange();
  theta_ = normalized * 2.0 * M_PI;
  // range is [0..2*pi)
  if (theta_ == 2.0 * M_PI) {
    theta_ = 0.0;
  }

  CHECK_GE(theta_, 0.0);
  CHECK_LT(theta_, 2.0 * M_PI);
}

double CyclicEncodable::getValue() const
{
  double normalized = theta_ / (2.0 * M_PI);
  double value = (normalized * getRange()) + rangeMin_;
  // TODO: std::clamp with C++ 17
  // .... C++ took 19 years to standardize, another 19 years to add clamp!!!!
  // Precautionary clamp to ensure return range is [rangeMin..rangeMax)
  value = std::max(rangeMin_, std::min(value, rangeMax_));
  // Limit to range: [rangeMin..rangeMax)
  if (value == rangeMax_) {
    value == rangeMin_;
  }
  return value;
}

void CyclicEncodable::encodeVector(
    std::vector<float>* feature_vec, const size_t index) const
{
  CHECK_LT(index + 1, feature_vec->size())
      << "Feature vector needs space for 2 floats from index";
  (*feature_vec)[index] = sin(theta_);
  (*feature_vec)[index+1] = cos(theta_);
}

void CyclicEncodable::decodeVector(
    const std::vector<float>& feature_vec, const size_t index) 
{
  CHECK_LT(index + 1, feature_vec.size())
      << "Feature vector needs space for 2 floats from index";
  float sinTheta = feature_vec[index];
  float cosTheta = feature_vec[index+1];
  theta_ = atan2(sinTheta, cosTheta);
  // Keep in range: [0..2*pi)
  if (theta_ < 0.0) {
    theta_ += (2.0 * M_PI);
  }
  CHECK_GE(theta_, 0.0);
  CHECK_LE(theta_, 2.0 * M_PI);
}

std::ostream & operator<<(std::ostream &os, const CyclicEncodable& cyclicObj)
{
  return os << std::fixed << std::setprecision(2)
      << "value: " << std::right << std::setw(6) << cyclicObj.getValue()
      << ", theta: " << std::right << std::setw(3) << cyclicObj.theta_
      // No alignment needed on range as likely same values across many objects
      << ", range: [" << cyclicObj.rangeMin_
      << " .. " << cyclicObj.rangeMax_
      << ")";
}
