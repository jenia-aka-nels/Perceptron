#pragma once
#include "mnist.h"
#include "functions.h"

class Perceptron
{
private:
	int n; // input size
	int k; // output size
	int s; // hidden layer size
	float speed;
	int number_of_images;
	int image_size;
	int number_of_labels;
	unsigned char** dataset;
	unsigned char* labels;

	double* w0; // n*s
	double* w1; // s*k
	double* deltaw0; // n*s
	double* deltaw1; // s*k

	MNIST* mnist;

	double* train(unsigned char* input, char output);
	bool test(unsigned char* input, char output);
	void shuffle();
public:
	Perceptron(string datapath, int _s, float speed);
	void Train();
	void Test();

};