#ifndef CYCLIC_ENCODABLE_H
#define CYCLIC_ENCODABLE_H

#include <vector>
#include <cstddef>  // std::size_t
#include <sstream>  // std::ostream
    
// Represent a continuous cyclic variable, encodeable for machine learning:
// * Encoded vector represenation is continuous, including when wrapping
// * Useful for angles, time or other variables which wrap
// Quantitatively validated for detection of angles:
// https://stats.stackexchange.com/a/218547
class CyclicEncodable
{
public:
  // E.g. value = 13.50, rangeMax = 24 (1.30pm on 24 hr clock)
  CyclicEncodable(const double value, const double rangeMax);

  // E.g. value = 90, rangeMin = -180, rangeMax = +180 (rotation angle)
  CyclicEncodable(
      const double value, const double rangeMin, const double rangeMax);

  // Custom copy constructor and assignment operator are required as
  // default methods can't set const values (rangeMin_ and rangeMax_)
  CyclicEncodable(const CyclicEncodable& src);
  CyclicEncodable& operator=(const CyclicEncodable& other);

  // get and set range is [rangeMin..rangeMax)
  double getValue() const;
  // CHECK fails setting value outside of range
  void setValue(double value);

  double getRange() const { return rangeMax_ - rangeMin_; }
  double getRangeMin() const { return rangeMin_; }
  double getRangeMax() const { return rangeMax_; }

  // Encode vector representation at index in vector, 2 floats written
  // Allows writing to vector that may contain other features, e.g. bbox
  void encodeVector(
      std::vector<float>* feature_vec, const std::size_t index) const;

  // Decode vector representation at index in vector, 2 floats read
  void decodeVector(
      const std::vector<float>& feature_vec, const std::size_t index);

private:
  // theta_ range is [0..2*pi), maps linearly to [rangeMin..rangeMax)
  double theta_;
  const double rangeMin_;
  const double rangeMax_;

  friend std::ostream& operator<<(std::ostream &os, const CyclicEncodable& cyclicObj);
};

std::ostream& operator<<(std::ostream &os, const CyclicEncodable& cyclicObj);

#endif // CYCLIC_ENCODABLE_H
