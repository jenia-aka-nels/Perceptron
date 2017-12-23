#pragma once
#include "headers.h"
using namespace std;

class MNIST
{
private:
	string trainimagespath;
	string trainlabelspath;
	string testimagespath;
	string testlabelspath;
	unsigned char** read_mnist_images(string full_path, int& number_of_images, int& image_size);
	unsigned char* read_mnist_labels(string full_path, int& number_of_labels);
public:
	MNIST(string path);

	unsigned char** read_mnist_train_images(int& number_of_images, int& image_size);
	unsigned char** read_mnist_test_images(int& number_of_images, int& image_size);
	unsigned char* read_mnist_train_labels(int& number_of_labels);
	unsigned char* read_mnist_test_labels(int& number_of_labels);
};