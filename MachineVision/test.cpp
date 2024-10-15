#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <random>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;
using namespace cv;

void addGaussianNoise(cv::Mat& image, double mean, double sigma) {
    cv::Mat noise(image.size(), image.type());
    cv::randn(noise, mean, sigma); // 高斯噪声
    image += noise;
}

void gaussianBlur(cv::Mat& image, int kernelSize) {
    cv::Mat temp = image.clone();
    int radius = kernelSize / 2;
    std::vector<double> kernel(kernelSize * kernelSize);

    // Generate Gaussian kernel
    double sigma = 1.0;
    double sum = 0.0;
    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {
            double value = exp(-(x * x + y * y) / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);
            kernel[(y + radius) * kernelSize + (x + radius)] = value;
            sum += value;
        }
    }

    // Normalize Gaussian kernel
    for (auto& val : kernel) {
        val /= sum;
    }

    // Apply Gaussian blur
    if (image.channels() == 3) {  // For 3-channel images
        for (int y = radius; y < image.rows - radius; y++) {
            for (int x = radius; x < image.cols - radius; x++) {
                cv::Vec3d newPixel = cv::Vec3d(0, 0, 0);
                for (int ky = -radius; ky <= radius; ky++) {
                    for (int kx = -radius; kx <= radius; kx++) {
                        newPixel += kernel[(ky + radius) * kernelSize + (kx + radius)] *
                            image.at<cv::Vec3b>(y + ky, x + kx);
                    }
                }
                image.at<cv::Vec3b>(y, x) = cv::Vec3b(cv::saturate_cast<uchar>(newPixel[0]),
                    cv::saturate_cast<uchar>(newPixel[1]),
                    cv::saturate_cast<uchar>(newPixel[2]));
            }
        }
    }
    else if (image.channels() == 1) {  // For single-channel images
        for (int y = radius; y < image.rows - radius; y++) {
            for (int x = radius; x < image.cols - radius; x++) {
                double newPixel = 0.0;
                for (int ky = -radius; ky <= radius; ky++) {
                    for (int kx = -radius; kx <= radius; kx++) {
                        newPixel += kernel[(ky + radius) * kernelSize + (kx + radius)] *
                            image.at<uchar>(y + ky, x + kx);
                    }
                }
                image.at<uchar>(y, x) = cv::saturate_cast<uchar>(newPixel);
            }
        }
    }
}


//NL-means 
void nonlocalMeansFilter(Mat& src, Mat& dest, int templeteWindowSize, int searchWindowSize, double h, double sigma = 0.0)
{

    if (dest.empty())
        dest = Mat::zeros(src.size(), src.type());

    const int tr = templeteWindowSize >> 1;                 //邻域中心位置
    const int sr = searchWindowSize >> 1;                   //搜索域中心位置
    const int bb = sr + tr;                                 //增加边界
    const int D = searchWindowSize * searchWindowSize;      //搜索域元素个数
    const int H = D / 2 + 1;                                //搜索域中心点位置
    const double div = 1.0 / (double)D;                     //均匀分布时，搜索域中的每个点的权重大小
    const int tD = templeteWindowSize * templeteWindowSize; //邻域中的元素个数
    const double tdiv = 1.0 / (double)(tD);                 //均匀分布时，搜索域中的每个点的权重大小
    Mat im;
    copyMakeBorder(src, im, bb, bb, bb, bb, cv::BORDER_DEFAULT);
    //计算权重
    vector<double> weight(256 * 256 * src.channels());
    double* w = &weight[0];
    const double gauss_sd = (sigma == 0.0) ? h : sigma;                             //高斯标准差
    double gauss_color_coeff = -(1.0 / (double)(src.channels())) * (1.0 / (h * h)); //高斯颜色系数
    int emax = 0;


    //w[i]保存方差，即邻域平均欧氏距离对应的高斯加权权重，供后面计算出欧式距离后调用
    for (int i = 0; i < 256 * 256 * src.channels(); i++)
    {
        double v = std::exp(max(i - 2.0 * gauss_sd * gauss_sd, 0.0) * gauss_color_coeff);
        w[i] = v;
        if (v < 0.001)
        {
            emax = i;
            break;
        }
    }
    for (int i = emax; i < 256 * 256 * src.channels(); i++)
        w[i] = 0.0;

    if (src.channels() == 3)//3通道
    {
        const int cstep = (int)im.step - templeteWindowSize * 3;
        const int csstep = (int)im.step - searchWindowSize * 3;
        #pragma omp parallel for
        for (int j = 0; j < src.rows; j++)
        { //j for rows
            uchar* d = dest.ptr(j);
            int* ww = new int[D];       //D 为搜索域中的元素数量，ww用于记录搜索域每个点的邻域方差
            double* nw = new double[D]; //根据方差大小高斯加权归一化后的权重
            for (int i = 0; i < src.cols; i++)
            { //i for cols
                double tweight = 0.0;
                //search loop
                uchar* tprt = im.data + im.step * (sr + j) + 3 * (sr + i);
                uchar* sptr2 = im.data + im.step * j + 3 * i;
                for (int l = searchWindowSize, count = D - 1; l--;)
                {
                    uchar* sptr = sptr2 + im.step * (l);
                    for (int k = searchWindowSize; k--;)
                    {
                        //templete loop
                        int e = 0;
                        uchar* t = tprt;
                        uchar* s = sptr + 3 * k;
                        for (int n = templeteWindowSize; n--;)
                        {
                            for (int m = templeteWindowSize; m--;)
                            {
                                // computing color L2 norm
                                e += (s[0] - t[0]) * (s[0] - t[0]) + (s[1] - t[1]) * (s[1] - t[1]) + (s[2] - t[2]) * (s[2] - t[2]); //L2 norm
                                s += 3;
                                t += 3;
                            }
                            t += cstep;
                            s += cstep;
                        }
                        const int ediv = e * tdiv;
                        ww[count--] = ediv;
                        //计算权重
                        tweight += w[ediv];
                    }
                }
                //权重归一化
                if (tweight == 0.0)
                {
                    for (int z = 0; z < D; z++)
                        nw[z] = 0;
                    nw[H] = 1;
                }
                else
                {
                    double itweight = 1.0 / (double)tweight;
                    for (int z = 0; z < D; z++)
                        nw[z] = w[ww[z]] * itweight;
                }
                double r = 0.0, g = 0.0, b = 0.0;
                uchar* s = im.ptr(j + tr);
                s += 3 * (tr + i);
                for (int l = searchWindowSize, count = 0; l--;)
                {
                    for (int k = searchWindowSize; k--;)
                    {
                        r += s[0] * nw[count];
                        g += s[1] * nw[count];
                        b += s[2] * nw[count++];
                        s += 3;
                    }
                    s += csstep;
                }
                d[0] = saturate_cast<uchar>(r);//防止颜色溢出
                d[1] = saturate_cast<uchar>(g);
                d[2] = saturate_cast<uchar>(b);
                d += 3;
            } //i
            delete[] ww;
            delete[] nw;
            cout << j << "---block for all ---->" << src.rows << endl;
        } //j
    }
    else if (src.channels() == 1)//对应于单通道图
    {
        const int cstep = (int)im.step - templeteWindowSize; //在邻域比较时，从邻域的上一行末尾跳至下一行开头
        const int csstep = (int)im.step - searchWindowSize;  //搜索域循环中，从搜索域的上一行末尾跳至下一行开头
        #pragma omp parallel for
        //下面两层嵌套循环：遍历每个图片的像素点
        for (int j = 0; j < src.rows; j++)
        {
            uchar* d = dest.ptr(j);
            int* ww = new int[D];       //D 为搜索域中的元素数量，ww用于记录搜索域每个点的邻域方差
            double* nw = new double[D]; //根据方差大小高斯加权归一化后的权重
            for (int i = 0; i < src.cols; i++)
            {
                //下面两层嵌套循环：遍历像素点（i，j）的搜索域点
                double tweight = 0.0;
                uchar* tprt = im.data + im.step * (sr + j) + (sr + i); //sr 为搜索域中心距
                uchar* sptr2 = im.data + im.step * j + i;
                for (int l = searchWindowSize, count = D - 1; l--;)
                {
                    uchar* sptr = sptr2 + im.step * (l);
                    for (int k = searchWindowSize; k--;)
                    {
                        //下面两层嵌套循环：对于每个像素点（i，j）的搜索域点的邻域进行比较
                        int e = 0; //累计方差
                        uchar* t = tprt;
                        uchar* s = sptr + k;
                        for (int n = templeteWindowSize; n--;)
                        {
                            for (int m = templeteWindowSize; m--;)
                            {
                                // computing color L2 norm
                                e += (*s - *t) * (*s - *t);
                                s++;
                                t++;
                            }
                            t += cstep;
                            s += cstep;
                        }
                        const int ediv = e * tdiv; //tdiv 搜索域均一分布权重大小
                        ww[count--] = ediv;
                        //get weighted Euclidean distance
                        tweight += w[ediv];
                    }
                }
                //weight normalization权重归一化
                if (tweight == 0.0)
                {
                    for (int z = 0; z < D; z++)
                        nw[z] = 0;
                    nw[H] = 1;
                }
                else
                {
                    double itweight = 1.0 / (double)tweight;
                    for (int z = 0; z < D; z++)
                        nw[z] = w[ww[z]] * itweight;
                }
                double v = 0.0;
                uchar* s = im.ptr(j + tr);
                s += (tr + i);
                for (int l = searchWindowSize, count = 0; l--;)
                {
                    for (int k = searchWindowSize; k--;)
                    {
                        v += *(s++) * nw[count++];
                    }
                    s += csstep;
                }
                *(d++) = saturate_cast<uchar>(v);
            } //i
            delete[] ww;
            delete[] nw;
        } //j
    }
}

void calcPSF(Mat& outputImg, Size filterSize, int len, double theta)
{
    Mat h(filterSize, CV_32F, Scalar(0));
    Point point(filterSize.width / 2, filterSize.height / 2);
    ellipse(h, point, Size(0, cvRound(float(len) / 2.0)), 90.0 - theta,
        0, 360, Scalar(255), FILLED);
    Scalar summa = sum(h);
    outputImg = h / summa[0];
    Mat tmp;
    normalize(outputImg, tmp, 1, 0, NORM_MINMAX);
    imshow("psf", tmp);
}
void fftshift(const Mat& inputImg, Mat& outputImg)
{
    outputImg = inputImg.clone();
    int cx = outputImg.cols / 2;
    int cy = outputImg.rows / 2;
    Mat q0(outputImg, Rect(0, 0, cx, cy));
    Mat q1(outputImg, Rect(cx, 0, cx, cy));
    Mat q2(outputImg, Rect(0, cy, cx, cy));
    Mat q3(outputImg, Rect(cx, cy, cx, cy));
    Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}
void filter2DFreq(const Mat& inputImg, Mat& outputImg, const Mat& H)
{
    Mat planes[2] = { Mat_<float>(inputImg.clone()), Mat::zeros(inputImg.size(), CV_32F) };
    Mat complexI;
    merge(planes, 2, complexI);
    dft(complexI, complexI, DFT_SCALE);
    Mat planesH[2] = { Mat_<float>(H.clone()), Mat::zeros(H.size(), CV_32F) };
    Mat complexH;
    merge(planesH, 2, complexH);
    Mat complexIH;
    mulSpectrums(complexI, complexH, complexIH, 0);
    idft(complexIH, complexIH);
    split(complexIH, planes);
    outputImg = planes[0];
}
void calcWnrFilter(const Mat& input_h_PSF, Mat& output_G, double nsr)
{
    Mat h_PSF_shifted;
    fftshift(input_h_PSF, h_PSF_shifted);
    Mat planes[2] = { Mat_<float>(h_PSF_shifted.clone()), Mat::zeros(h_PSF_shifted.size(), CV_32F) };
    Mat complexI;
    merge(planes, 2, complexI);
    dft(complexI, complexI);
    split(complexI, planes);
    Mat denom;
    pow(abs(planes[0]), 2, denom);
    denom += nsr;
    divide(planes[0], denom, output_G);
}
void edgetaper(const Mat& inputImg, Mat& outputImg, double gamma, double beta)
{
    int Nx = inputImg.cols;
    int Ny = inputImg.rows;
    Mat w1(1, Nx, CV_32F, Scalar(0));
    Mat w2(Ny, 1, CV_32F, Scalar(0));
    float* p1 = w1.ptr<float>(0);
    float* p2 = w2.ptr<float>(0);
    float dx = float(2.0 * CV_PI / Nx);
    float x = float(-CV_PI);
    for (int i = 0; i < Nx; i++)
    {
        p1[i] = float(0.5 * (tanh((x + gamma / 2) / beta) - tanh((x - gamma / 2) / beta)));
        x += dx;
    }
    float dy = float(2.0 * CV_PI / Ny);
    float y = float(-CV_PI);
    for (int i = 0; i < Ny; i++)
    {
        p2[i] = float(0.5 * (tanh((y + gamma / 2) / beta) - tanh((y - gamma / 2) / beta)));
        y += dy;
    }
    Mat w = w2 * w1;
    multiply(inputImg, w, outputImg);
}

int main() {
    cv::Mat originalImage = cv::imread("Resources\\input512.jpg", IMREAD_GRAYSCALE);
    cv::Mat degradedImage = originalImage.clone();
    cv::Mat restoredImage;

    // 降质处理
    addGaussianNoise(degradedImage, 0, 25);
    //gaussianBlur(degradedImage, 5);

	// 去模糊处理
    //Rect roi(0, 0, degradedImage.cols & -2, degradedImage.rows & -2); // 裁剪图像尺寸为偶数
    //degradedImage = degradedImage(roi);
    //int LEN = 50;      // 模糊长度
    //double THETA = 30; // 模糊角度
    //double snr = 8000; // 信噪比
    //Mat h_PSF, Hw, imgOut;
    //calcPSF(h_PSF, roi.size(), LEN, THETA);
    //calcWnrFilter(h_PSF, Hw, 1.0 / snr);
    //degradedImage.convertTo(degradedImage, CV_32F);
    //filter2DFreq(degradedImage, imgOut, Hw);
    //imgOut.convertTo(imgOut, CV_8U);
    //normalize(imgOut, imgOut, 0, 255, NORM_MINMAX);
    //imshow("Deblurred Image", imgOut);
    //imwrite("deblurred_result.jpg", imgOut);
    //waitKey(0);


    // 复原处理
	//cv::fastNlMeansDenoisingColored(degradedImage, restoredImage, 10, 10, 7, 21);
	nonlocalMeansFilter(degradedImage, restoredImage, 7, 21, 10, 0.0);  // 非局部均值滤波去噪


    // 显示结果
    cv::imshow("Original Image", originalImage);
    cv::imshow("Degraded Image", degradedImage);
    cv::imshow("Restored Image", restoredImage);
    cv::waitKey(0);

    // 保存结果
    cv::imwrite("Outputs\\degraded_image.jpg", degradedImage);
    cv::imwrite("Outputs\\restored_image.jpg", restoredImage);

    return 0;
}
