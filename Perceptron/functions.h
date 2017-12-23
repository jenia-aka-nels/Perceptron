#pragma once
#include "headers.h"

class Functions
{
public:
	void softmax(int k, double* a)
	{
		double sum = 0;
		for (int i = 0; i < k; i++)
		{
			sum += exp(a[i]);
		}
		for (int i = 0; i < k; i++)
		{
			a[i] = exp(a[i]) / sum;
		}
	}

	void sigm(int s, double* a)
	{
		for (int i = 0; i < s; i++)
		{
			a[i] = 1 / (1 + exp(-2 * a[i]));
		}
	}
};
