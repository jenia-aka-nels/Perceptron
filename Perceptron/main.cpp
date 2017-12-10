#include <stdio.h>
#include <stdlib.h> 
#include <math.h>
#include <iostream>
#include <fstream>

using namespace std;

double* train(float speed, int n, int s, int k, unsigned char* input, unsigned char output, double* w0, double* w1, double* deltaw0, double* deltaw1);
bool test(int n, int s, int k, unsigned char* input, unsigned char output, double* w0, double* w1);
void sigm(int s, double* a);
void softmax(int k, double* result);
unsigned char** read_mnist_images(string full_path, int& number_of_images, int& image_size);
unsigned char* read_mnist_labels(string full_path, int& number_of_labels);

int main(int argc, char** argv)
{
	srand(0);
	float speed = 0.1f;
	int s = 300;
	string datapath = "C:\\Documents\\dataset";
	if (argc > 1)
	{
		datapath = argv[1];
		for (int i = 0; i < datapath.length(); i++)
		{
			if (datapath[i] == '\\') datapath[i] = '/';
		}

	}
	if (argc > 2)
	{
		s = atoi(argv[2]);
	}

	if (argc > 3)
	{
		speed = atof(argv[3]);
	}

	int n; // input size
	int k; // output size

	int number_of_images;
	int image_size;
	int number_of_labels;
	unsigned char** dataset; // n*inN
	unsigned char* labels; // k*outN
	unsigned char* input;
	unsigned char output;

	double* w0; // n*s
	double* w1; // s*k
	double* deltaw0; // n*s
	double* deltaw1; // s*k

	//============INIT=================================
	int randmax = 45;

	number_of_images = 0;
	image_size = 0;
	number_of_labels = 0;

	string imagespath = datapath + "\\train-images.idx3-ubyte";
	string labelspath = datapath + "\\train-labels.idx1-ubyte";

	dataset = read_mnist_images(imagespath, number_of_images, image_size);
	labels = read_mnist_labels(labelspath, number_of_labels);
	printf("DATA LOADED");
	k = 10;
	n = image_size;

	input = new unsigned char[n];

	w0 = new double[n*s];
	w1 = new double[s*k];
	deltaw0 = new double[n*s];
	deltaw1 = new double[s*k];
	for (int i = 0; i < n*s; i++)
	{
		w0[i] = (rand() % randmax) / 1000.0;
		deltaw0[i] = 0;
	}
	for (int i = 0; i < s*k; i++)
	{
		w1[i] = (rand() % randmax) / 1000.0;
		deltaw1[i] = 0;
	}

	//============TRAINING=================================

	double* result;
	for (int i = 0; i < number_of_images; i++)
	{
		result = train(speed, n, s, k, dataset[i], labels[i], w0, w1, deltaw0, deltaw1);
		if (i % 2000 == 0)
		{
			printf("\n%d\t", i);
			/*for (int l = 0; l < k; l++)
			{
				printf("%.3f\t", result[l]);
			}*/
		}
	}
	printf("\n%d\t", number_of_images);
	/*for (int l = 0; l < k; l++)
	{
		printf("%.3f\t", result[l]);
	}*/
	printf("\nTRAINED\n");
	delete dataset;
	delete labels;

	//============TESTING=================================
	int wronganswers = 0;

	imagespath = datapath + "\\t10k-images.idx3-ubyte";
	labelspath = datapath + "\\t10k-labels.idx1-ubyte";

	dataset = read_mnist_images(imagespath, number_of_images, image_size);
	labels = read_mnist_labels(labelspath, number_of_labels);
	for (int i = 0; i < number_of_images; i++)
	{
		if (!test(n, s, k, dataset[i], labels[i], w0, w1)) wronganswers++;
		if (i % 2000 == 0)
		{
			printf("\n%d\t", i);
		}
	}
	printf("\nERROR: %.2f\n", (wronganswers / (float)number_of_images) * 100);
	getchar();
	return 0;
}


double* train(float speed, int n, int s, int k, unsigned char* input, unsigned char output, double* w0, double* w1, double* deltaw0, double* deltaw1)
{
	double* hidden = new double[s];
	int* out = new int[k];
	double* result = new double[k];
	double* deltaout = new double[k];
	double* deltahid = new double[s];

	for (int i = 0; i < k; i++)
	{
		if (i == output) out[i] = 1; else out[i] = 0;
	}

	for (int j = 0; j < s; j++)
	{
		hidden[j] = 0;
		for (int i = 0; i < n; i++)
		{
			hidden[j] += input[i] * w0[i*s + j]/ 255;
		}
	}
	sigm(s, hidden);
	for (int j = 0; j < k; j++)
	{
		result[j] = 0;
		for (int i = 0; i < s; i++)
		{
			result[j] += hidden[i] * w1[i*k + j];
		}
	}
	softmax(k, result);

	for (int i = 0; i < k; i++)
	{
		deltaout[i] = result[i] - out[i];
	}
	for (int i = 0; i < s; i++)
	{
		deltahid[i] = 0;
		for (int j = 0; j < k; j++)
		{
			deltahid[i] += deltaout[j] * w1[i*k + j] * (hidden[i] * (1 - hidden[i]));
		}
	}
	bool tmp = false;
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < s; j++)
		{
			deltaw0[i*s + j] = speed * deltahid[j] * input[i]/255;
			w0[i*s + j] -= deltaw0[i*s + j];
		}
	}
	for (int i = 0; i < s; i++)
	{
		for (int j = 0; j < k; j++)
		{
			deltaw1[i*k + j] = speed * deltaout[j] * hidden[i];
			w1[i*k + j] -= deltaw1[i*k + j];
		}
	}
	delete hidden;
	delete result;
	delete out;
	delete deltaout;
	delete deltahid;
	return result;
}

bool test(int n, int s, int k, unsigned char* input, unsigned char output, double* w0, double* w1)
{
	double* hidden = new double[s];
	double* result = new double[k];

	for (int j = 0; j < s; j++)
	{
		hidden[j] = 0;
		for (int i = 0; i < n; i++)
		{
			hidden[j] += input[i] * w0[i*s + j]/ 255;
		}
	}
	sigm(s, hidden);
	for (int j = 0; j < k; j++)
	{
		result[j] = 0;
		for (int i = 0; i < s; i++)
		{
			result[j] += hidden[i] * w1[i*k + j];
		}
	}

	softmax(k, result);

	int maxI = 0; 
	double maxR = result[0];
	for (int i = 1; i < k; i++)
	{
		if (result[i] > maxR)
		{
			maxI = i;
			maxR = result[i];
		}
	}
	delete hidden;
	delete result;
	if (maxI != output) return false;
	return true;
}

void softmax(int k, double* a)
{
	double sum = 0;
	for (int i = 0; i < k; i++)
	{
		sum += exp(a[i]);
	}
	for (int i = 0; i < k; i++)
	{
		a[i] = exp(a[i])/sum;
	}
}

void sigm(int s, double* a)
{
	for (int i = 0; i < s; i++)
	{
		a[i] = 1 / (1 + exp(-2 * a[i]));
	}
}

unsigned char** read_mnist_images(string full_path, int& number_of_images, int& image_size) {
	auto reverseInt = [](int i) {
		unsigned char c1, c2, c3, c4;
		c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
		return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
	};

	ifstream file(full_path, ios::binary);
	
	if (file.is_open()) {
		int magic_number = 0, n_rows = 0, n_cols = 0;

		file.read((char *)&magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);

		if (magic_number != 2051) throw runtime_error("Invalid MNIST image file!");

		file.read((char *)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
		file.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
		file.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

		image_size = n_rows * n_cols;

		unsigned char** _dataset = new unsigned char*[number_of_images];
		for (int i = 0; i < number_of_images; i++) {
			_dataset[i] = new unsigned char[image_size];
			file.read((char *)_dataset[i], image_size);
		}
		return _dataset;
	}
	else {
		throw runtime_error("Cannot open file `" + full_path + "`!");
	}
}

unsigned char* read_mnist_labels(string full_path, int& number_of_labels) {
	auto reverseInt = [](int i) {
		unsigned char c1, c2, c3, c4;
		c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
		return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
	};

	typedef unsigned char uchar;

	ifstream file(full_path, ios::binary);

	if (file.is_open()) {
		int magic_number = 0;
		file.read((char *)&magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);

		if (magic_number != 2049) throw runtime_error("Invalid MNIST label file!");

		file.read((char *)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);

		uchar* _dataset = new uchar[number_of_labels];
		for (int i = 0; i < number_of_labels; i++) {
			file.read((char*)&_dataset[i], 1);
		}
		return _dataset;
	}
	else {
		throw runtime_error("Unable to open file `" + full_path + "`!");
	}
}