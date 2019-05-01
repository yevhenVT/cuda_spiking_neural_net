#include "hvcRA.cuh"
#include "make_network.h"
#include <iostream>
#include <fstream>
#include <string>

#define NUM_THREADS_IN_BLOCK 1024
#define BUFFER_SIZE 2000

void write_data(double* buffer, int num_iter, const char *filename)
{
	std::ofstream out;
	out.open(filename, std::ios::out | std::ios::binary);

	out.write(reinterpret_cast<char*>(&num_iter), sizeof(int));

	for (int i = 0; i < num_iter; i++)
	{
		//std::cout << "time = " << buffer[i*3] 
		//		  << " Vs = "  << buffer[i*3+1]
		//		  << " Vd = "  << buffer[i*3+2] << std::endl; 

		out.write(reinterpret_cast<char*>(&buffer[i*3]), sizeof(double));
		out.write(reinterpret_cast<char*>(&buffer[i*3+1]), sizeof(double));
		out.write(reinterpret_cast<char*>(&buffer[i*3+2]), sizeof(double));

	}

	out.close();
}

int main(int argc, char** argv)
{
	int N = 20000;
	double ampl_s = 0;
	double ampl_d = 0;
	std::string filename = "/storage/home/yzt116/ConcurrentMatrixComputation/mini_project/test.bin";
	
	if (argc > 1)
	{	
		ampl_s = atof(argv[1]);
		ampl_d = atof(argv[2]);
		filename = argv[3];

		std::cout << "ampl_s = " << ampl_s << " ampl_d = " << ampl_d << std::endl;
		std::cout << "filename = " << filename << std::endl;
	}

	// initialize arrays
	double *h_vars, *d_vars; 
	double *h_buffer, *d_buffer;
	bool *h_record, *d_record;

	h_vars = new double[N*14];
	h_buffer = new double [BUFFER_SIZE*3];
	h_record = new bool[N];

	// initialize variables
	for (int i = 0; i < N; i++)
	{
		h_vars[i] = 0; // time
		h_vars[i + 1*N] = -79.97619025; // Vs
		h_vars[i + 2*N] = 0.01101284; // n
		h_vars[i + 3*N] = 0.9932845; // h
		h_vars[i + 4*N] = -79.97268759; // Vd
		h_vars[i + 5*N] = 0.00055429; // r
		h_vars[i + 6*N] = 0.00000261762353; // c
		h_vars[i + 7*N] = 0.01689572; // Ca
		h_vars[i + 8*N] = 0; // Gexc_d
		h_vars[i + 9*N] = 0; // Gexc_s
		h_vars[i + 10*N] = 0; // Ginh_d
		h_vars[i + 11*N] = 0; // Ginh_s
		h_vars[i + 12*N] = 0; // Id
		h_vars[i + 13*N] = 0; // Is

		h_record[i] = false;
	}

	for (int i = 0; i < 3*BUFFER_SIZE; i++)
	{
		h_buffer[i] = 0.0;
	}
	cudaMalloc(&d_vars, N*14*sizeof(double));
	cudaMalloc(&d_record, N*sizeof(bool));
	cudaMalloc(&d_buffer, BUFFER_SIZE*3*sizeof(double));
	
	cudaMemcpy(d_vars, h_vars, N*14*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_record, h_record, N*sizeof(bool), cudaMemcpyHostToDevice);
	cudaMemcpy(d_buffer, h_buffer, BUFFER_SIZE*3*sizeof(double), cudaMemcpyHostToDevice);

	delete[] h_buffer; delete[] h_record; delete[] h_vars;

	double timestep = 0.01;
	double trial = 100;
	int num_iter = static_cast<int>(trial / timestep);

	int num_blocks = N / NUM_THREADS_IN_BLOCK + 1;

	int neuron_to_record = 1;
	int training = 0;

	double *big_buffer;
	
	big_buffer = new double[3*(num_iter / BUFFER_SIZE)* BUFFER_SIZE];

	std::cout << "Before setting record\n" << std::endl;
	
	set_record<<<num_blocks, NUM_THREADS_IN_BLOCK >>>(d_record, neuron_to_record);

	int step = 0;

	std::cout << "Before starting iterations\n" << std::endl;

	for (int i = 0; i < num_iter; i++)
	{
		//std::cout << "time = " << static_cast<double>(i) * timestep << std::endl;

	 	calculate_next_step_RA<<<num_blocks, NUM_THREADS_IN_BLOCK >>>(d_vars, N, timestep, d_record, d_buffer, step, ampl_s, ampl_d, training);

		cudaDeviceSynchronize();

		if ( (i+1) % BUFFER_SIZE == 0)
		{
			step = 0;
			
			//print_buffer<<<num_blocks, NUM_THREADS_IN_BLOCK >>>(d_buffer, BUFFER_SIZE);
			
			int ind = ( (i+1)/BUFFER_SIZE - 1 ) * BUFFER_SIZE * 3;
			
			cudaMemcpy(&big_buffer[ind], d_buffer, BUFFER_SIZE*3*sizeof(double), cudaMemcpyDeviceToHost);
			
			//std::cout << "ind = " << ind 
			//		  << " time = " << big_buffer[ind]
			//		  << " Vs = " << big_buffer[ind + 1]
			//		  << " Vd = " << big_buffer[ind + 2] << std::endl;
		}
		else
			step += 1;
	}

	write_data(big_buffer, (num_iter/BUFFER_SIZE) * BUFFER_SIZE, filename.c_str());

	delete[] big_buffer;
	cudaFree(d_buffer); cudaFree(d_vars); cudaFree(d_record);

}
