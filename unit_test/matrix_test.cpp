// MUST include this
#include <catch2/catch.hpp>

// File being tested
//#include <El/core/Matrix.hpp>

// Other includes
#include <El.hpp>
#include <hydrogen/meta/TypeTraits.hpp>

TEMPLATE_TEST_CASE(
  "Testing Sequential Matrix","[seq][matrix]",
  float, double)
{
    using value_type = TestType;
    using matrix_type = El::Matrix<value_type, El::Device::CPU>;
    using size_type = typename matrix_type::size_type;

    auto const zero_size = hydrogen::TypeTraits<size_type>::Zero();

    GIVEN("An empty matrix")
    {
        auto mat = matrix_type{};

        THEN ("The matrix has dimension 0x0 with nonzero LDim.")
        {
            CHECK(mat.Height() == zero_size);
            CHECK(mat.Width() == zero_size);
            CHECK(mat.LDim() > zero_size);
            CHECK(mat.MemorySize() == zero_size);
            CHECK(mat.Contiguous());
        }

        WHEN ("The matrix is resized")
        {
            mat.Resize(7,11);

            THEN ("The change is reflected in the metadata.")
            {
                CHECK(mat.Height() == size_type{7});
                CHECK(mat.Width() == size_type{11});
                CHECK(mat.LDim() == size_type{7});
                CHECK(mat.MemorySize() == mat.LDim()*mat.Width());
                CHECK(mat.Contiguous());
            }
            AND_WHEN ("The matrix is shrunk")
            {
                mat.Resize(5,12);

                THEN ("The change is reflected in the metadata.")
                {
                    CHECK(mat.Height() == size_type{5});
                    CHECK(mat.Width() == size_type{12});
                    CHECK(mat.LDim() == size_type{5});
                    CHECK(mat.MemorySize() == 77);
                    CHECK(mat.Contiguous());
                }
            }
            AND_WHEN ("The matrix is expanded")
            {
                mat.Resize(11,13);
                THEN ("The change is reflected in the metadata.")
                {
                    CHECK(mat.Height() == size_type{11});
                    CHECK(mat.Width() == size_type{13});
                    CHECK(mat.LDim() == size_type{11});
                    CHECK(mat.MemorySize() == 11*13);
                    CHECK(mat.Contiguous());
                }
            }
            AND_WHEN ("The matrix is resized with a large leading dimension")
            {
                mat.Resize(7,13,11);
                THEN ("The change is reflected in the metadata.")
                {
                    CHECK(mat.Height() == size_type{7});
                    CHECK(mat.Width() == size_type{13});
                    CHECK(mat.LDim() == size_type{11});
                    CHECK(mat.MemorySize() == 11*13);
                    CHECK_FALSE(mat.Contiguous());
                }
            }
            AND_WHEN ("The matrix is resized with an invalid leading dimension")
            {
                THEN ("The resize throws and metadata is unchanged.")
                {
                    CHECK_THROWS(mat.Resize(17,19,2));
                    CHECK(mat.Height() == size_type{7});
                    CHECK(mat.Width() == size_type{11});
                    CHECK(mat.LDim() == size_type{7});
                    CHECK(mat.MemorySize() == 7*11);
                    CHECK(mat.Contiguous());
                }
            }
        }
    }
}
