#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;


vector<vector<int>> return_blocks(Mat img, int block_size);

double findecc(InputArray templateImage,
			   InputArray inputImage,
			   InputOutputArray warpMatrix,
			   int motionType,
			   TermCriteria criteria,
			   InputArray inputMask,
			   int gaussFiltSize);

static void image_jacobian_homo_ECC(const Mat& src1, const Mat& src2,
	const Mat& src3, const Mat& src4,
	const Mat& src5, Mat& dst);

static void image_jacobian_euclidean_ECC(const Mat& src1, const Mat& src2,
	const Mat& src3, const Mat& src4,
	const Mat& src5, Mat& dst);

static void image_jacobian_affine_ECC(const Mat& src1, const Mat& src2,
	const Mat& src3, const Mat& src4,
	Mat& dst);

static void image_jacobian_translation_ECC(const Mat& src1, const Mat& src2, Mat& dst);

static void project_onto_jacobian_ECC(const Mat& src1, const Mat& src2, Mat& dst);

static void update_warping_matrix_ECC(Mat& map_matrix, const Mat& update, const int motionType);
