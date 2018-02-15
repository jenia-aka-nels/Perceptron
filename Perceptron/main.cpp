#include "perceptron.h"

int main(int argc, char** argv)
{
	srand(0);
	float speed = 0.1f;
	int s = 300;
	int epoch = 1;
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

	if (argc > 4)
	{
		epoch = atof(argv[4]);
	}

	Perceptron* perceptron = new Perceptron(datapath, s, speed);

	for (int i = 0; i < epoch; i++)
	{
		perceptron->Train();
		perceptron->Test();
	}

	return 0;
}