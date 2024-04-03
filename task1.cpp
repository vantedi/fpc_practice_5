//Задание 1
#include <iostream>
#include <opencv2/opencv.hpp>
#include <mpi.h>

using namespace std;
using namespace cv;

const int width = 800;
const int height = 600;
const double x_min = -2.0;
const double x_max = 1.0;
const double y_min = -1.5;
const double y_max = 1.5;
const int max_iterations = 1000;
const double sq_radius = 4.0;

Scalar getColor(int iterations, double real, double imag) {
    if (iterations == max_iterations) {
        return Scalar(0, 0, 0);
    }
    double t = (double)iterations / max_iterations;
    int hue = (int)(360 * t);
    double dist = sqrt(real * real + imag * imag);
    int value = (int)(255 * dist);
    return Scalar(hue, 255, value); 
}

int mandelbrot(double real, double imag) {
    double zReal = 0.0, zImag = 0.0;
    int iterations = 0;

    while (zReal * zReal + zImag * zImag < sq_radius && iterations < max_iterations) {
        double temp = zReal * zReal - zImag * zImag + real;
        zImag = 2.0 * zReal * zImag + imag;
        zReal = temp;
        iterations++;
    }

    return iterations;
}

int main(int argc, char** argv) {
    setlocale(LC_ALL, "Russian");

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Mat image(height, width, CV_8UC3, Scalar(0, 0, 0));

    int rowsPerProcess = height / size;
    int startRow = rank * rowsPerProcess;
    int endRow = startRow + rowsPerProcess;

    for (int y = startRow; y < endRow; ++y) {
        double imag = y_min + y * (y_max - y_min) / height;
        for (int x = 0; x < width; ++x) {
            double real = x_min + x * (x_max - x_min) / width;
            int iterations = mandelbrot(real, imag);
            Scalar color = getColor(iterations, real, imag);
            image.at<Vec3b>(y, x) = Vec3b(color[0], color[1], color[2]);
        }
    }

    if (rank != 0) {
        MPI_Send(image.data, height * width * 3, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
    }
    else {
        for (int i = 1; i < size; ++i) {
            Mat recvImage(height / size, width, CV_8UC3);
            MPI_Recv(recvImage.data, height / size * width * 3, MPI_BYTE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            recvImage.copyTo(image.rowRange(i * rowsPerProcess, (i + 1) * rowsPerProcess));
        }

        imshow("фрактал Мандельброта", image);
        waitKey(0);
    }

    MPI_Finalize();

    return 0;
}
