// ECC_algorithm.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//

#include <iostream>
#include "opencv2/opencv.hpp"
#include <iostream>
#include <string>
#include "FindEcc.h"

using namespace std;
using namespace cv;

Mat get_gradient(Mat gray_img);

Mat image_alignment(Mat im, int block_size);

vector<Mat> full_channels(vector<Mat> channels, Mat im, int height, int width);

int main(int argc, char **args)
{                               
	Mat im = imread(args[1]);

	im = image_alignment(im, 64);
	
	imshow("out.bmp", im);
	waitKey(0);

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

	vector<Vec3b> blocks = return_blocks(im, block_size, 20);

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

	int number_of_iterations = 16;
	double termination_eps = 1e-10;

	TermCriteria criteria(TermCriteria::COUNT + TermCriteria::EPS,
		number_of_iterations, termination_eps);

	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < blocks.size(); j++) {
			int x = blocks[j][0];
			int y = blocks[j][1];
			double cc = findecc(get_gradient(Mat(big_channels[2], Rect(x, y, block_size, block_size))), get_gradient(Mat(big_channels[i], Rect(x, y, block_size, block_size))), warp_matrix, warp_mode, criteria, Mat(), 5);
		}
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



// Запуск программы: CTRL+F5 или меню "Отладка" > "Запуск без отладки"
// Отладка программы: F5 или меню "Отладка" > "Запустить отладку"

// Советы по началу работы 
//   1. В окне обозревателя решений можно добавлять файлы и управлять ими.
//   2. В окне Team Explorer можно подключиться к системе управления версиями.
//   3. В окне "Выходные данные" можно просматривать выходные данные сборки и другие сообщения.
//   4. В окне "Список ошибок" можно просматривать ошибки.
//   5. Последовательно выберите пункты меню "Проект" > "Добавить новый элемент", чтобы создать файлы кода, или "Проект" > "Добавить существующий элемент", чтобы добавить в проект существующие файлы кода.
//   6. Чтобы снова открыть этот проект позже, выберите пункты меню "Файл" > "Открыть" > "Проект" и выберите SLN-файл.
