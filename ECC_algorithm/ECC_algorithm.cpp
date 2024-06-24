#include "opencv2/opencv.hpp"
#include <iostream>
#include <string>
#include "FindEcc.h"
#include <time.h>

using namespace std;
using namespace cv;

int show_warp_mat(Mat warp_matrix);

Mat get_gradient(Mat gray_img);

Mat image_alignment(Mat im, int block_size);

int compare_methods_of_image_alignment(Mat im, int block_size);

vector<Mat> full_channels(vector<Mat> channels, Mat im, int height, int width);

int main(int argc, char **args)
{                               
	Mat im = imread(args[1]);
	int blocks_size = 128;
	//im = image_alignment(im, blocks_size);
	compare_methods_of_image_alignment(im, blocks_size);
	imwrite("out.png", im);
	waitKey(0);

	return 0;
}

int show_warp_mat(Mat warp_matrix) {
	for (int h = 0; h < warp_matrix.size().height; h++) {
		for (int w = 0; w < warp_matrix.size().width; w++) {
			cout << warp_matrix.at<float>(h, w) << " ";
		}
		cout << "\n";
	}
	return 0;
}

Mat get_gradient(Mat gray_img) {
	Mat gradient_x, gradient_y;
	Mat abs_gradient_x, abs_gradient_y;
	int scale = 1;
	int delta = 0;
	int dataDepth = CV_32FC1;
	Sobel(gray_img, gradient_x, dataDepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(gradient_x, abs_gradient_x);

	Sobel(gray_img, gradient_y, dataDepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(gradient_y, abs_gradient_y);

	Mat gradient;
	addWeighted(abs_gradient_x, 0.5, abs_gradient_y, 0.5, 0, gradient);

	return gradient;
}

vector<Mat> full_channels(vector<Mat> channels, Mat im, int height, int width) {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			Vec3b pixel = im.at<Vec3b>(i, j);
			channels[0].at<unsigned char>(i, j) = pixel[0];
			channels[1].at<unsigned char>(i, j) = pixel[1];
			channels[2].at<unsigned char>(i, j) = pixel[2];
		}
	}
	return channels;
}

Mat image_alignment(Mat im, int block_size) {
	Size sz = im.size();
	int height = sz.height;
	int width = sz.width;

	vector<vector<int>> blocks = return_blocks(im, block_size);
	vector<Mat> big_channels;
	big_channels.push_back(Mat(height, width, CV_8UC1));
	big_channels.push_back(Mat(height, width, CV_8UC1));
	big_channels.push_back(Mat(height, width, CV_8UC1));
	big_channels = full_channels(big_channels, im, height, width);

	vector<Mat> aligned_channels;
	for (int i = 0; i < 2; i++) {
		aligned_channels.push_back(Mat(height, width, CV_8UC1));
	}
	aligned_channels.push_back(big_channels[2].clone());

	const int warp_mode = MOTION_TRANSLATION;

	Mat warp_matrix;

	if (warp_mode == MOTION_HOMOGRAPHY) {
		warp_matrix = Mat::eye(3, 3, CV_32F);
	}
	else {
		warp_matrix = Mat::eye(2, 3, CV_32F);
	}

	int number_of_iterations = 200;
	double termination_eps = 1e-10;

	TermCriteria criteria(TermCriteria::COUNT + TermCriteria::EPS,
		number_of_iterations, termination_eps);

	for (int i = 0; i < 2; i++) {
		vector<float> warp_matrixes_0;
		vector<float> warp_matrixes_1;
		for (int j = 0; j < blocks.size(); j++) {
			warp_matrix = Mat::eye(2, 3, CV_32F);
			int x = blocks[j][0];
			int y = blocks[j][1];
			double cc = findecc(get_gradient(Mat(big_channels[2], Rect(y, x, block_size, block_size))), get_gradient(Mat(big_channels[i], Rect(y, x, block_size, block_size))), warp_matrix, warp_mode, criteria, Mat(), 5);
			warp_matrixes_0.push_back(warp_matrix.clone().at<float>(0, 2));
			warp_matrixes_1.push_back(warp_matrix.clone().at<float>(1, 2));
			//show_warp_mat(warp_matrixes[j]);
		}
		auto m = warp_matrixes_0.begin() + warp_matrixes_0.size() / 2;
		nth_element(warp_matrixes_0.begin(), m, warp_matrixes_0.end());
		m = warp_matrixes_1.begin() + warp_matrixes_1.size() / 2;
		nth_element(warp_matrixes_1.begin(), m, warp_matrixes_1.end());
		warp_matrix.at<float>(0, 2) = warp_matrixes_0[warp_matrixes_0.size() / 2];
		warp_matrix.at<float>(1, 2) = warp_matrixes_1[warp_matrixes_1.size() / 2];
		show_warp_mat(warp_matrix);
		//double cc = findecc(get_gradient(big_channels[2]), get_gradient(big_channels[i]), warp_matrix, warp_mode, criteria, Mat(), 5);

		if (warp_mode != MOTION_HOMOGRAPHY) {
			warpAffine(big_channels[i], aligned_channels[i], warp_matrix, aligned_channels[0].size(), INTER_LINEAR + WARP_INVERSE_MAP);
		}
		else {
			warpPerspective(big_channels[i], aligned_channels[i], warp_matrix, aligned_channels[0].size(), INTER_LINEAR + WARP_INVERSE_MAP);
		}
	}
	Mat im_aligned;
	merge(aligned_channels, im_aligned);
	return im_aligned;
}

int compare_methods_of_image_alignment(Mat im, int block_size) {
	Size sz = im.size();
	int height = sz.height;
	int width = sz.width;

	vector<vector<int>> blocks = return_blocks(im, block_size);
	cout << blocks.size();
	vector<Mat> big_channels;
	big_channels.push_back(Mat(height, width, CV_8UC1));
	big_channels.push_back(Mat(height, width, CV_8UC1));
	big_channels.push_back(Mat(height, width, CV_8UC1));
	big_channels = full_channels(big_channels, im, height, width);

	vector<Mat> aligned_channels;
	for (int i = 0; i < 2; i++) {
		aligned_channels.push_back(Mat(height, width, CV_8UC1));
	}
	aligned_channels.push_back(big_channels[2].clone());

	const int warp_mode = MOTION_TRANSLATION;

	Mat warp_matrix;

	int number_of_iterations = 16;
	double termination_eps = 1e-10;

	TermCriteria criteria(TermCriteria::COUNT + TermCriteria::EPS,
		number_of_iterations, termination_eps);

	for (int i = 0; i < 2; i++) {
		cout << "\nnow iteration\n\n";
		clock_t start = clock();
		vector<float> warp_matrixes_0;
		vector<float> warp_matrixes_1;
		warp_matrix = Mat::eye(2, 3, CV_32F);
		for (int j = 0; j < blocks.size(); j++) {
			warp_matrix = Mat::eye(2, 3, CV_32F);
			int x = blocks[j][0];
			int y = blocks[j][1];
			double cc = findecc(get_gradient(Mat(big_channels[2], Rect(y, x, block_size, block_size))), get_gradient(Mat(big_channels[i], Rect(y, x, block_size, block_size))), warp_matrix, warp_mode, criteria, Mat(), 5);
			warp_matrixes_0.push_back(warp_matrix.clone().at<float>(0, 2));
			warp_matrixes_1.push_back(warp_matrix.clone().at<float>(1, 2));
			//show_warp_mat(warp_matrixes[j]);
		}
		auto m = warp_matrixes_0.begin() + warp_matrixes_0.size() / 2;
		nth_element(warp_matrixes_0.begin(), m, warp_matrixes_0.end());
		m = warp_matrixes_1.begin() + warp_matrixes_1.size() / 2;
		nth_element(warp_matrixes_1.begin(), m, warp_matrixes_1.end());
		warp_matrix.at<float>(0, 2) = warp_matrixes_0[warp_matrixes_0.size() / 2];
		warp_matrix.at<float>(1, 2) = warp_matrixes_1[warp_matrixes_1.size() / 2];
		clock_t end = clock();
		double seconds = (double)(end - start) / CLOCKS_PER_SEC;
		cout << "Method with blocks, it worked " << seconds << "\n";
		show_warp_mat(warp_matrix);
		if (warp_mode == MOTION_HOMOGRAPHY) {
			warp_matrix = Mat::eye(3, 3, CV_32F);
		}
		else {
			warp_matrix = Mat::eye(2, 3, CV_32F);
		}
		start = clock();
		double cc = findecc(get_gradient(big_channels[2]), get_gradient(big_channels[i]), warp_matrix, warp_mode, criteria, Mat(), 5);
		end = clock();
		seconds = (double)(end - start) / CLOCKS_PER_SEC;
		cout << "Method without blocks, it worked " << seconds << "\n";
		show_warp_mat(warp_matrix);
		if (warp_mode != MOTION_HOMOGRAPHY) {
			warpAffine(big_channels[i], aligned_channels[i], warp_matrix, aligned_channels[0].size(), INTER_LINEAR + WARP_INVERSE_MAP);
		}
		else {
			warpPerspective(big_channels[i], aligned_channels[i], warp_matrix, aligned_channels[0].size(), INTER_LINEAR + WARP_INVERSE_MAP);
		}
	}
	return 0;
}
