#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <ctime>  

using namespace cv;
using namespace std;

Mat computeDisparityNCC(const Mat& imgL, const Mat& imgR, int numDisparities = 64, int blockSize = 5) {
    Mat disparity = Mat::zeros(imgL.size(), CV_32F);

    int halfBlockSize = blockSize / 2;
    for (int y = halfBlockSize; y < imgL.rows - halfBlockSize; y++) {
        for (int x = halfBlockSize; x < imgL.cols - halfBlockSize; x++) {
            float maxNCC = -1.0f;
            int bestDisparity = 0;

            for (int d = 0; d < numDisparities; d++) {
                if (x - d < halfBlockSize) break;

                Rect leftRect(x - halfBlockSize, y - halfBlockSize, blockSize, blockSize);
                Rect rightRect(x - d - halfBlockSize, y - halfBlockSize, blockSize, blockSize);

                Mat blockL = imgL(leftRect);
                Mat blockR = imgR(rightRect);

                // 计算每个块的均值
                float meanL = mean(blockL)[0];
                float meanR = mean(blockR)[0];

                // 计算归一化相关性
                Mat nccL = blockL - meanL;
                Mat nccR = blockR - meanR;

                float numerator = sum(nccL.mul(nccR))[0];
                float denominator = sqrt(sum(nccL.mul(nccL))[0] * sum(nccR.mul(nccR))[0]);

                float ncc = numerator / (denominator + 1e-10); // 防止除零

                if (ncc > maxNCC) {
                    maxNCC = ncc;
                    bestDisparity = d;
                }
            }
            disparity.at<float>(y, x) = (float)bestDisparity;
        }
    }

    Mat dispNCCVis;
    normalize(disparity, dispNCCVis, 0, 255, NORM_MINMAX, CV_8U);

    return dispNCCVis;
}

Mat computeDisparitySAD(const Mat& imgL, const Mat& imgR, int numDisparities = 64, int blockSize = 5) {
    Mat disparity = Mat::zeros(imgL.size(), CV_32F);

    int halfBlockSize = blockSize / 2;
    for (int y = halfBlockSize; y < imgL.rows - halfBlockSize; y++) {
        for (int x = halfBlockSize; x < imgL.cols - halfBlockSize; x++) {
            float minSAD = FLT_MAX;
            int bestDisparity = 0;

            for (int d = 0; d < numDisparities; d++) {
                if (x - d < halfBlockSize) break;

                Rect leftRect(x - halfBlockSize, y - halfBlockSize, blockSize, blockSize);
                Rect rightRect(x - d - halfBlockSize, y - halfBlockSize, blockSize, blockSize);

                Mat blockL = imgL(leftRect);
                Mat blockR = imgR(rightRect);

                float sad = sum(abs(blockL - blockR))[0];

                if (sad < minSAD) {
                    minSAD = sad;
                    bestDisparity = d;
                }
            }
            disparity.at<float>(y, x) = (float)bestDisparity;
        }
    }

    Mat dispSADVis;
    normalize(disparity, dispSADVis, 0, 255, NORM_MINMAX, CV_8U);

    return dispSADVis;
}

Mat computeDisparityBM(const Mat& imgL, const Mat& imgR, int numDisparities = 16 * 5, int blockSize = 21) {
    Ptr<StereoBM> stereoBM = StereoBM::create();
    stereoBM->setNumDisparities(numDisparities);
    stereoBM->setBlockSize(blockSize);

    Mat disparityBM;
    stereoBM->compute(imgL, imgR, disparityBM);

    Mat dispBMVis;
    normalize(disparityBM, dispBMVis, 0, 255, NORM_MINMAX, CV_8U);

    return dispBMVis;
}

Mat computeDisparitySGBM(const Mat& imgL, const Mat& imgR, int numDisparities = 16 * 5, int blockSize = 5) {
    Ptr<StereoSGBM> stereoSGBM = StereoSGBM::create();
    stereoSGBM->setMinDisparity(0);
    stereoSGBM->setNumDisparities(numDisparities);
    stereoSGBM->setBlockSize(blockSize);
    stereoSGBM->setP1(8 * 1 * blockSize * blockSize);
    stereoSGBM->setP2(32 * 1 * blockSize * blockSize);
    stereoSGBM->setMode(StereoSGBM::MODE_SGBM);

    Mat disparitySGBM;
    stereoSGBM->compute(imgL, imgR, disparitySGBM);

    Mat dispSGBMVis;
    normalize(disparitySGBM, dispSGBMVis, 0, 255, NORM_MINMAX, CV_8U);

    return dispSGBMVis;
}

int main() {
    Mat imgL = imread("./tsukuba/tsukuba_l.png", IMREAD_GRAYSCALE);
    Mat imgR = imread("./tsukuba/tsukuba_r.png", IMREAD_GRAYSCALE);

    if (imgL.empty() || imgR.empty()) {
        cout << "图像读入错误!" << endl;
        return -1;
    }

    clock_t start, end;
    double time_NCC, time_SAD, time_BM, time_SGBM;

    start = clock();
    Mat disparityNCC = computeDisparityNCC(imgL, imgR);
    end = clock();
    time_NCC = static_cast<double>(end - start) / CLOCKS_PER_SEC;
    cout << "Time of NCC: " << time_NCC << endl;

    start = clock();
    Mat disparitySAD = computeDisparitySAD(imgL, imgR);
    end = clock();
    time_SAD = static_cast<double>(end - start) / CLOCKS_PER_SEC;
    cout << "Time of SAD: " << time_SAD << endl;

    start = clock();
    Mat disparityBM = computeDisparityBM(imgL, imgR);
    end = clock();
    time_BM = static_cast<double>(end - start) / CLOCKS_PER_SEC;
    cout << "Time of BM: " << time_BM << endl;

    start = clock();
    Mat disparitySGBM = computeDisparitySGBM(imgL, imgR);
    end = clock();
    time_SGBM = static_cast<double>(end - start) / CLOCKS_PER_SEC;
    cout << "Time of SGBM: " << time_SGBM << endl;

    imshow("Disparity - NCC", disparityNCC);
    imshow("Disparity - SAD", disparitySAD);
    imshow("Disparity - StereoBM", disparityBM);
    imshow("Disparity - StereoSGBM", disparitySGBM);
    waitKey(0);

	imwrite("Outputs\\ParallaxEstimate\\NCC.png", disparityNCC);
	imwrite("Outputs\\ParallaxEstimate\\SAD.png", disparitySAD);
	imwrite("Outputs\\ParallaxEstimate\\StereoBM.png", disparityBM);
	imwrite("Outputs\\ParallaxEstimate\\StereoSGBM.png", disparitySGBM);


    return 0;
}
