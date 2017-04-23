#include "cyclic_encodable.h"

#include <cmath>
#include <limits>

#include "gtest/gtest.h"


namespace {

// The fixture for testing class CyclicEncodable.
class CyclicEncodableTest : public ::testing::Test {
 protected:
  // Objects declared here can be used by all tests for CyclicEncodable
};

TEST_F(CyclicEncodableTest, ConstructorTwoParam_valid) {
  // 1.30pm within 24 hour clock
  CyclicEncodable cyclicObj(13.5, 24.0);
  EXPECT_DOUBLE_EQ(0.0, cyclicObj.getRangeMin());
  EXPECT_DOUBLE_EQ(24.0, cyclicObj.getRangeMax());
  EXPECT_DOUBLE_EQ(13.5, cyclicObj.getValue());
}

TEST_F(CyclicEncodableTest, ConstructorTwoParam_rangeInverted) {
  ASSERT_DEATH({
    CyclicEncodable cyclicObj(13.5, -24.0);
  // Skip escaping periods in regexes for readability
  }, "Check failed: rangeMin_ < rangeMax_ \\(0 vs. -24\\) ");
}

TEST_F(CyclicEncodableTest, ConstructorTwoParam_rangeTooSmall) {
  ASSERT_DEATH({
    CyclicEncodable cyclicObj(13.5, 0.5);
  }, "Check failed: getRange\\(\\) > kSuspiciouslySmallRange \\(0.5 vs. 0.9\\)");
}

TEST_F(CyclicEncodableTest, ConstructorThreeParam) {
  // 90 degrees with range: -180..180
  CyclicEncodable cyclicObj(90, -180.0, 180.0);
  EXPECT_DOUBLE_EQ(-180.0, cyclicObj.getRangeMin());
  EXPECT_DOUBLE_EQ(180.0, cyclicObj.getRangeMax());
  EXPECT_DOUBLE_EQ(90.0, cyclicObj.getValue());
}

TEST_F(CyclicEncodableTest, ConstructorThreeParam_rangeInverted) {
  ASSERT_DEATH({
    CyclicEncodable cyclicObj(13.5, -16.0, -24.0);
  }, "Check failed: rangeMin_ < rangeMax_ \\(-16 vs. -24\\) ");
}
    
TEST_F(CyclicEncodableTest, ConstructorThreeParam_rangeTooSmall) {
  ASSERT_DEATH({
    CyclicEncodable cyclicObj(13.5, 13.2, 14.0);
  }, "Check failed: getRange\\(\\) > kSuspiciouslySmallRange \\(0.8 vs. 0.9\\)");
}

TEST_F(CyclicEncodableTest, CopyConstructor) {
  CyclicEncodable cyclicObj(90, -180.0, 180.0);

  CyclicEncodable copyObj(cyclicObj);
  EXPECT_DOUBLE_EQ(-180.0, copyObj.getRangeMin());
  EXPECT_DOUBLE_EQ(180.0, copyObj.getRangeMax());
  EXPECT_DOUBLE_EQ(90.0, copyObj.getValue());
}

TEST_F(CyclicEncodableTest, AssignmentOperator) {
  CyclicEncodable angle1(90, -180.0, 180.0);
  CyclicEncodable angle2(50, -180.0, 180.0);

  angle1 = angle2;
  EXPECT_DOUBLE_EQ(-180.0, angle1.getRangeMin());
  EXPECT_DOUBLE_EQ(180.0, angle1.getRangeMax());
  EXPECT_DOUBLE_EQ(50.0, angle1.getValue());
}

TEST_F(CyclicEncodableTest, AssignmentOperator_rangeMinDifferent) {
  CyclicEncodable angle1(5, -180.0, 180.0);
  CyclicEncodable angle2(5, -10.0, 180.0);
  ASSERT_DEATH({
    angle1 = angle2;
  }, "CyclicEncodable assignment to object with different rangeMin");
}

TEST_F(CyclicEncodableTest, AssignmentOperator_rangeMaxDifferent) {
  CyclicEncodable angle1(5, -180.0, 180.0);
  CyclicEncodable angle2(5, -180.0, 10.0);
  ASSERT_DEATH({
    angle1 = angle2;
  }, "CyclicEncodable assignment to object with different rangeMax");
}

TEST_F(CyclicEncodableTest, SetValue_Simple) {
  CyclicEncodable angle(90, -180.0, 180.0);
  angle.setValue(-90.0);
  EXPECT_DOUBLE_EQ(-90.0, angle.getValue());
}

TEST_F(CyclicEncodableTest, SetValue_WrapAtLimit) {
  CyclicEncodable angle(20.0, -180.0, 180.0);
  angle.setValue(180.0);
  EXPECT_DOUBLE_EQ(-180.0, angle.getValue());
}

TEST_F(CyclicEncodableTest, SetValue_Limits) {
  CyclicEncodable angle(20.0, -180.0, 180.0);
  EXPECT_DOUBLE_EQ(20.0, angle.getValue());
  angle.setValue(-180.0);
  EXPECT_DOUBLE_EQ(-180.0, angle.getValue());
  angle.setValue(179.9);
  EXPECT_DOUBLE_EQ(179.9, angle.getValue());
}

TEST_F(CyclicEncodableTest, SetValue_GTRangeMax) {
  CyclicEncodable angle1(90, -180.0, 180.0);
  ASSERT_DEATH({
    angle1.setValue(200.0);
  }, "Check failed: value .\\= rangeMax_ \\(200 vs. 180\\)");
}

TEST_F(CyclicEncodableTest, SetValue_LTRangeMin) {
  CyclicEncodable angle1(90, -180.0, 180.0);
  ASSERT_DEATH({
    angle1.setValue(-200.0);
  }, "Check failed: value .\\= rangeMin_ \\(-200 vs. -180\\)");
}

TEST_F(CyclicEncodableTest, GetValue_Simple) {
  CyclicEncodable angle(20.0, -180.0, 180.0);
  angle.setValue(50.0);
  EXPECT_DOUBLE_EQ(50.0, angle.getValue());
}

// Repeats same test as SetValue_WrapAtLimit
TEST_F(CyclicEncodableTest, GetValue_WrapAtLimit) {
  CyclicEncodable angle(20.0, -180.0, 180.0);
  angle.setValue(180.0);
  EXPECT_DOUBLE_EQ(-180.0, angle.getValue());
}

TEST_F(CyclicEncodableTest, GetValue_Limits) {
  CyclicEncodable angle(0.0, -180.0, 180.0);

  angle.setValue(179.999);
  EXPECT_DOUBLE_EQ(179.999, angle.getValue());

  angle.setValue(-180.0);
  EXPECT_DOUBLE_EQ(-180.0, angle.getValue());
}

TEST_F(CyclicEncodableTest, GetRange) {
  CyclicEncodable angle(20.0, -180.0, 180.0);
  EXPECT_DOUBLE_EQ(360.0, angle.getRange());
}

TEST_F(CyclicEncodableTest, GetRange_DifferentRange) {
  CyclicEncodable angle(20.0, 0.0, 24.0);
  EXPECT_DOUBLE_EQ(24.0, angle.getRange());
}

TEST_F(CyclicEncodableTest, GetRangeMin) {
  CyclicEncodable angle(20.0, -180.0, 180.0);
  EXPECT_DOUBLE_EQ(180.0, angle.getRangeMax());
}

TEST_F(CyclicEncodableTest, GetRangeMax) {
  CyclicEncodable angle(20.0, -180.0, 180.0);
  EXPECT_DOUBLE_EQ(-180.0, angle.getRangeMin());
}

TEST_F(CyclicEncodableTest, EncodeVector_Simple) {
  CyclicEncodable angle(50.0, -180.0, 180.0);
  std::vector<float> vec(6, 0.0);

  angle.encodeVector(&vec, 4);
  EXPECT_EQ(6, vec.size());
  EXPECT_FLOAT_EQ(0.0, vec[0]);
  EXPECT_FLOAT_EQ(0.0, vec[1]);
  EXPECT_FLOAT_EQ(0.0, vec[2]);
  EXPECT_FLOAT_EQ(0.0, vec[3]);
  EXPECT_FLOAT_EQ(-0.76604444, vec[4]);
  EXPECT_FLOAT_EQ(-0.64278764, vec[5]);
}

TEST_F(CyclicEncodableTest, EncodeVector_CloseNearWrap) {
  // Ensure that distant values near wrap are mapped to close vectors
  // Not value = 180.0 as fmod in setValue can wrap that
  CyclicEncodable angleBefore(-179.999, -180.0, 180.0);
  CyclicEncodable angleAfter(179.999, -180.0, 180.0);
  std::vector<float> vecBefore(2);
  std::vector<float> vecAfter(2);

  angleBefore.encodeVector(&vecBefore, 0);
  angleAfter.encodeVector(&vecAfter, 0);
  
  EXPECT_NEAR(vecBefore[0], vecAfter[0], 0.0001);
  EXPECT_NEAR(vecBefore[1], vecAfter[1], 0.0001);
}

TEST_F(CyclicEncodableTest, EncodeVector_InvalidIndex) {
  CyclicEncodable angle(50.0, -180.0, 180.0);
  std::vector<float> vec(6, 0.0);
  ASSERT_DEATH({
    angle.encodeVector(&vec, 5);
  }, "Check failed: index \\+ 1 < feature_vec->size\\(\\) "
     "\\(6 vs. 6\\) Feature vector needs space for 2 floats from index");
}

TEST_F(CyclicEncodableTest, DecodeVector_Simple) {
  CyclicEncodable angle(50.0, -180.0, 180.0);
  std::vector<float> vec(6, 0.0);
  vec[4] = -0.76604444;
  vec[5] = -0.64278764;

  angle.decodeVector(vec, 4);
  EXPECT_FLOAT_EQ(50.0, angle.getValue());
}

TEST_F(CyclicEncodableTest, DecodeVector_TimeRange) {
  CyclicEncodable encoder(13.5, 24.0);
  CyclicEncodable decoder(0.0, 24.0);
  std::vector<float> vec(2);

  encoder.encodeVector(&vec, 0);
  EXPECT_FLOAT_EQ(-0.38268343, vec[0]);
  EXPECT_FLOAT_EQ(-0.9238795, vec[1]);

  decoder.decodeVector(vec, 0);
  EXPECT_FLOAT_EQ(13.5, decoder.getValue());
}

TEST_F(CyclicEncodableTest, DecodeVector_AllQuadrants) {
  CyclicEncodable encoder(0.0, -180.0, 180.0);
  CyclicEncodable decoder(0.0, -180.0, 180.0);
  std::vector<float> vec(2);

  encoder.setValue(-180.0);
  encoder.encodeVector(&vec, 0);
  decoder.decodeVector(vec, 0);
  EXPECT_FLOAT_EQ(-180.0, decoder.getValue());

  // Testing at 45 degrees could still pass if sin and cos were swapped
  // Therefore test slightly on and off diagonal for better logic testing
  encoder.setValue(-140.0);
  encoder.encodeVector(&vec, 0);
  decoder.decodeVector(vec, 0);
  EXPECT_FLOAT_EQ(-140.0, decoder.getValue());

  encoder.setValue(-90.0);
  encoder.encodeVector(&vec, 0);
  decoder.decodeVector(vec, 0);
  EXPECT_FLOAT_EQ(-90.0, decoder.getValue());

  encoder.setValue(-45.0);
  encoder.encodeVector(&vec, 0);
  decoder.decodeVector(vec, 0);
  EXPECT_FLOAT_EQ(-45.0, decoder.getValue());

  encoder.setValue(0.0);
  encoder.encodeVector(&vec, 0);
  decoder.decodeVector(vec, 0);
  EXPECT_FLOAT_EQ(0.0, decoder.getValue());

  encoder.setValue(53.0);
  encoder.encodeVector(&vec, 0);
  decoder.decodeVector(vec, 0);
  EXPECT_FLOAT_EQ(53.0, decoder.getValue());

  encoder.setValue(90.0);
  encoder.encodeVector(&vec, 0);
  decoder.decodeVector(vec, 0);
  EXPECT_FLOAT_EQ(90.0, decoder.getValue());

  encoder.setValue(137.0);
  encoder.encodeVector(&vec, 0);
  decoder.decodeVector(vec, 0);
  EXPECT_FLOAT_EQ(137.0, decoder.getValue());

  encoder.setValue(179.9);
  encoder.encodeVector(&vec, 0);
  decoder.decodeVector(vec, 0);
  EXPECT_FLOAT_EQ(179.9, decoder.getValue());
}

// TODO: error in angle should be proportional to L1 loss for vector.
// Due to cartesian representation of a polar coordinate, when theta_ is
// near a diagonal, the vector representation will change by a factor sqrt(2)
// more than when theta_ is near the axis. As a consequence, an error of 1
// degree near the axis will cause 1.414 times the L1 loss in the encoded
// vector, compared to the same 1 degree error near the diagonal.
TEST_F(CyclicEncodableTest, DecodeVector_VectorDiffProportionalToError) {
  CyclicEncodable cyclic1(0.0, -180.0, 180.0);
  CyclicEncodable cyclic2(0.0, -180.0, 180.0);
  std::vector<float> vec1(2);
  std::vector<float> vec2(2);

  float minError = std::numeric_limits<float>::max();
  float maxError = 0.0;
  for (int angle = -180; angle < 180; angle++) {
    cyclic1.setValue(angle);
    cyclic2.setValue(angle + 1);
    cyclic1.encodeVector(&vec1, 0);
    cyclic2.encodeVector(&vec2, 0);
    float diff = fabs(vec1[0] - vec2[0]) + fabs(vec1[1] - vec2[1]);
    minError = std::min(minError, diff);
    maxError = std::max(maxError, diff);
  }

  // TODO: when fixed, these two values should be almost equal
  EXPECT_FLOAT_EQ(0.017604696, minError);
  EXPECT_FLOAT_EQ(0.024681389, maxError);  // maxError ~= sqrt(2) * minError
}
    
TEST_F(CyclicEncodableTest, StreamMethod) {
  CyclicEncodable angle(50.0, -180.0, 180.0);
  std::stringstream buffer;

  buffer << angle;
  const std::string tmp = buffer.str();
  EXPECT_STREQ("value:  50.00, theta: 4.01, range: [-180.00 .. 180.00)", tmp.c_str());
}
    
}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
