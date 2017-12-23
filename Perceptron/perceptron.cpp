#include "perceptron.h"

using namespace std;

Perceptron::Perceptron(string datapath, int _s, float _speed)
{
	int randmax = 45; // (randmax/1000) - maximal initial weight
	k = 10;
	s = _s;
	speed = _speed;
	number_of_images = 0;
	image_size = 0;
	number_of_labels = 0;

	mnist = new MNIST(datapath);

	dataset = mnist->read_mnist_train_images(number_of_images, image_size);
	labels = mnist->read_mnist_train_labels(number_of_labels);
	printf("DATA LOADED");
	n = image_size;

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
}

double* Perceptron::train(unsigned char* input, char output)
{
	double* hidden = new double[s];
	int* out = new int[k];
	double* result = new double[k];
	double* deltaout = new double[k];
	double* deltahid = new double[s];
	Functions* func = new Functions();
	for (int i = 0; i < k; i++)
	{
		if (i == output) out[i] = 1; else out[i] = 0;
	}

	for (int j = 0; j < s; j++)
	{
		hidden[j] = 0;
		for (int i = 0; i < n; i++)
		{
			hidden[j] += input[i] * w0[i*s + j] / 255;
		}
	}
	func->sigm(s, hidden);
	for (int j = 0; j < k; j++)
	{
		result[j] = 0;
		for (int i = 0; i < s; i++)
		{
			result[j] += hidden[i] * w1[i*k + j];
		}
	}
	func->softmax(k, result);

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
			deltaw0[i*s + j] = speed * deltahid[j] * input[i] / 255;
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

void Perceptron::Train()
{
	double* result;
	for (int i = 0; i < number_of_images; i++)
	{
		result = train(dataset[i], labels[i]);
		if (i % 2000 == 0)
		{
			printf("\n%d\t", i);
		}
	}
	printf("\n%d\t", number_of_images);
	printf("\nTRAINED\n");
}

bool Perceptron::test(unsigned char* input, char output)
{
	double* hidden = new double[s];
	double* result = new double[k];
	Functions* func = new Functions();
	for (int j = 0; j < s; j++)
	{
		hidden[j] = 0;
		for (int i = 0; i < n; i++)
		{
			hidden[j] += input[i] * w0[i*s + j] / 255;
		}
	}
	func->sigm(s, hidden);
	for (int j = 0; j < k; j++)
	{
		result[j] = 0;
		for (int i = 0; i < s; i++)
		{
			result[j] += hidden[i] * w1[i*k + j];
		}
	}

	func->softmax(k, result);

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

void Perceptron::Test()
{
	int wronganswers = 0;

	delete dataset;
	delete labels;
	dataset = mnist->read_mnist_test_images(number_of_images, image_size);
	labels = mnist->read_mnist_test_labels(number_of_labels);

	for (int i = 0; i < number_of_images; i++)
	{
		if (!test(dataset[i], labels[i])) wronganswers++;
		if (i % 2000 == 0)
		{
			printf("\n%d\t", i);
		}
	}
	printf("\nERROR: %.2f\n", (wronganswers / (float)number_of_images) * 100);
	printf("ACCURACY: %.4f\n", 1 - (wronganswers / (float)number_of_images));
	delete dataset;
	delete labels;
}