//
// Uber, Inc. (c) 2016
//


#define EIGEN_USE_MKL_ALL

#include <Eigen/Core>

#include <chrono>
#include <iostream>
#include <ostream>
#include <vector>

#include <cassert>
#include <cmath>
#include <cstdint>

#include <time.h>
#define NOMINMAX
#include <windows.h>
#include <mkl.h>
#include <omp.h>


using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::string;
using std::vector;

#define Mx int(10000)
#define My int(Mx*0.8)
#define n int(Mx*0.6)

namespace
{
// Functions from http://stackoverflow.com/questions/17432502/how-can-i-measure-cpu-time-and-wall-clock-time-on-both-linux-windows
// to measure wall time and cpu time
	double get_cpu_time() {
		FILETIME a, b, c, d;
		if (GetProcessTimes(GetCurrentProcess(), &a, &b, &c, &d) != 0) {
			return
				(double)(d.dwLowDateTime | ((unsigned long long)d.dwHighDateTime << 32))*0.0000001;
		}
		else {
			return 0;
		}
	}
	double get_wall_time() {
		return (double)clock() / CLOCKS_PER_SEC;
	}

// Timer utility
class measure_time_and_print
{
public:
    measure_time_and_print(string name)
        : customized_name(std::move(name)), start_wall_time(get_wall_time()), start_cpu_time(get_cpu_time())
    {
    }

    ~measure_time_and_print()
    {
        std::cout << "Wall time for " << customized_name << " is " << get_wall_time() - start_wall_time << std::endl;
        std::cout << "CPU time for " << customized_name << " is " << get_cpu_time() - start_cpu_time << std::endl << std::endl;
    }
private:
    const double start_wall_time;
    const double start_cpu_time;
    const string customized_name;
};
}

template<typename type>
void matrix_cr_conv(type *ipmatrix, size_t dx, size_t dy) {
	int i;
	int j;
	type *opmatrix;
	opmatrix = (type *)malloc(sizeof(type)*dx*dy);
	for (i = 0; i < dy; i++) {
		for (j = 0; j < dx; j++) {
			opmatrix[i*dx + j] = ipmatrix[i + j*dy];
		}
	}
	for (i = 0; i < dy*dx; i++) {
		ipmatrix[i] = opmatrix[i];
	}
	free(opmatrix);
	return;
}

namespace math
{
// This is the baseline I would like to compare with
// This is a pure Eigen implementation which uses Eigen Matrix Object and its operations
// At compile time, I will use -O3 flag to invoke the Eigen's optimization.
// However, I did not enable openmp, in which case the performance could be further boosted
MatrixXd euclidean_distance_square_with_eigen(const MatrixXd &x, const MatrixXd &y)
{
    MatrixXd d(x.rows(), y.rows());
	VectorXd xx;
	VectorXd yy;

	{
		measure_time_and_print timer("square sum:");
		xx = x.array().square().rowwise().sum();
		yy = y.array().square().rowwise().sum();

	}

	{
		measure_time_and_print timer("dgemm:");
		d = (-2. * x) * y.transpose();
	}
	//for (int i = 0; i < 20; i++) {
	//	std::cout << d.data()[i] << std::endl;
	//}
	//system("pause");
	{
		measure_time_and_print timer("mat add");
		d = d.array().colwise() + xx.array();

		//for (int i = 0; i < 20; i++) {
		//	std::cout << d.data()[i] << std::endl;
		//}
		//system("pause");

		d = d.array().rowwise() + yy.transpose().array();
	}
	//for (int i = 0; i < 20; i++) {
	//	std::cout << d.data()[i] << std::endl;
	//}
	//system("pause");
    return d;
}

struct matrix
{
    uint64_t rows = 0;
    uint64_t cols = 0;
    vector<double> data; // Data stored in a column major manner

    explicit matrix(const MatrixXd &mat)
    {
        rows = mat.rows();
        cols = mat.cols();
        data.resize(rows * cols);
        Eigen::Map<Eigen::MatrixXd>(data.data(), rows, cols) = mat;
    }

    matrix(const uint64_t num_rows, const int64_t num_cols)
        : rows(num_rows), cols(num_cols)
    {
        data.resize(rows * cols);
    }

    matrix() = default;

    // Elementwise access
    double &operator()(const uint64_t row_index, const uint64_t col_index)
    {
        return data[row_index + col_index * rows];
    }

    double operator()(const uint64_t row_index, const uint64_t col_index) const
    {
        return data[row_index + col_index * rows];
    }

    // Elementwise access
    double &operator()(const uint64_t index)
    {
        assert(rows == 1 || cols == 1);
        return data[index];
    }

    double operator()(const uint64_t index) const
    {
        assert(rows == 1 || cols == 1);
        return data[index];
    }

    // Compare with a Eigen Matrix
    bool operator==(const MatrixXd &mat) const
    {
        if (mat.rows() != rows || mat.cols() != cols)
        {
            return false;
        }
        return Eigen::Map<const MatrixXd>(data.data(), rows, cols).isApprox(mat);
    }
};

inline std::ostream &operator<<(std::ostream &os, const math::matrix &mat)
{
    os << Eigen::Map<const MatrixXd>(mat.data.data(), mat.rows, mat.cols);
    return os;
}

// function needs to be optimized
matrix matrix_element_wise_square_and_rowwise_summation(const matrix &mat)
{
    matrix result(mat.rows, 1);

    // Initialization
    for (uint64_t index = 0; index < mat.rows; index++)
    {
        result(index) = 0;
    }
    // Computation
    for (uint64_t col_index = 0; col_index < mat.cols; col_index++)
    {
        for (uint64_t row_index = 0; row_index < mat.rows; row_index++)
        {
            result(row_index) += mat(row_index, col_index) * mat(row_index, col_index);
        }
    }
    return result;
}

// function needs to be optimized
matrix matrix_multiplication_with_second_being_transposed_and_element_wise_rescale(
    const matrix &a, const matrix &b, const double scale)
{
    assert(a.cols == b.cols);

    matrix result(a.rows, b.rows);

    for (uint64_t col_index = 0; col_index < result.cols; col_index++)
    {
        for (uint64_t row_index = 0; row_index < result.rows; row_index++)
        {
            result(row_index, col_index) = 0;
            for (uint64_t index = 0; index < a.cols; index++)
            {
                result(row_index, col_index) += scale * a(row_index, index) * b(col_index, index);
            }
        }
    }
    return result;
}

// function needs to be optimized
matrix &col_wise_element_wise_addition(const matrix &vec, matrix &mat)
{
    assert(vec.rows == 1 || vec.cols == 1);
    assert(vec.rows == mat.rows || vec.cols == mat.rows);

    for (uint64_t col_index = 0; col_index < mat.cols; col_index++)
    {
        for (uint64_t row_index = 0; row_index < mat.rows; row_index++)
        {
            mat(row_index, col_index) += vec(row_index);
        }
    }
    return mat;
}

// function needs to be optimized
matrix &row_wise_element_wise_addition(const matrix &vec, matrix &mat)
{
    assert(vec.rows == 1 || vec.cols == 1);
    assert(vec.rows == mat.cols || vec.cols == mat.cols);

    for (uint64_t col_index = 0; col_index < mat.cols; col_index++)
    {
        for (uint64_t row_index = 0; row_index < mat.rows; row_index++)
        {
            mat(row_index, col_index) += vec(col_index);
        }
    }
    return mat;
}

// The five utility functions shown above require optimization.
// We would like to make them run as fast as they can in a single thread.
// And we would also like to see they can be optimized to run in a multithreaded setting.
matrix euclidean_distance_square_with_loop(const matrix &x, const matrix &y)
{
    auto xx = matrix_element_wise_square_and_rowwise_summation(x);
    auto yy = matrix_element_wise_square_and_rowwise_summation(y);
    auto distance = matrix_multiplication_with_second_being_transposed_and_element_wise_rescale(x, y, -2.0);
    distance = col_wise_element_wise_addition(xx, distance);
    distance = row_wise_element_wise_addition(yy, distance);
    return distance;
}
}




double* MKL_Euclidean_distance(double* x,double* y)
{
	//double* x_ones, *y_ones;
	double* x_y_mul = (double*)MKL_malloc(Mx*My * sizeof(double), 64);


	//	x_ones = (double*)MKL_malloc(n*My * sizeof(double), 64);
	//	y_ones = (double*)MKL_malloc(Mx*n * sizeof(double), 64);
	//{
	//	measure_time_and_print timer("ones:");
	//


	//	for (int i = 0; i < n*My; i++)
	//	{
	//		x_ones[i] = 1;
	//	}
	//	for (int j = 0; j < Mx*n; j++)
	//	{
	//		y_ones[j] = 1;
	//	}
	//}
	//Dist = sqrt(A.^2*ones(size(B'))+ones(size(A))*(B').^2-2*A*B')


	{
		measure_time_and_print timer("dgemm:");
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Mx, My, n, -2, x, n, y, n, 0, x_y_mul, My);//-2*x*y'
	}

	//{
	//	measure_time_and_print timer("row square:");
	//	vdSqr(Mx*n, x, x); //x.^2
	//	vdSqr(My*n, y, y); //y.^2
	//}

	//double* x_E_sum = (double*)MKL_malloc(Mx*My * sizeof(double), 64);
	//double* y_E_sum = (double*)MKL_malloc(Mx*My * sizeof(double), 64);

	double* x_row_sqr_sum = (double*)MKL_malloc(Mx * 1 * sizeof(double), 64);
	double* y_row_sqr_sum = (double*)MKL_malloc(My * 1 * sizeof(double), 64);

	MKL_Set_Num_Threads(1);
	{
		measure_time_and_print timer("square sum:");
		//cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Mx, My, n, 1, x, n, x_ones, My, 0, x_E_sum, My);//x.^2*ones(size(y'))
		//cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Mx, My, n, 1, y_ones, n, y, n, 0, y_E_sum, My);//ones(size(x))*(y.^2)'
#pragma omp parallel for
		for (int i = 0; i != Mx; i++) {
			x_row_sqr_sum[i] = cblas_ddot(n, &x[i*n], 1, &x[i*n], 1);
		}
		for (int i = 0; i != My; i++) {
			y_row_sqr_sum[i] = cblas_ddot(n, &y[i*n], 1, &y[i*n], 1);
		}
	}


	MKL_free(x);
	MKL_free(y);

	//for (int i = 0; i < 20; i++) {
	//	std::cout << x_y_mul[i] << std::endl;
	//}
	//system("pause");

	//MKL_free(x_ones);
	//MKL_free(y_ones);
	{
		measure_time_and_print timer("mat add:");
#pragma omp parallel for
		for (int i = 0; i != Mx; i++) {
			cblas_daxpy(My, 1, &x_row_sqr_sum[i], 0, &x_y_mul[i*My], 1);

			cblas_daxpy(My, 1, y_row_sqr_sum, 1, &x_y_mul[i*My], 1);
		}

		//for (int i = 0; i < 20; i++) {
		//	std::cout << x_y_mul[i] << std::endl;
		//}
		//system("pause");


	}
	//for (int i = 0; i < 20; i++) {
	//	std::cout << x_y_mul[i] << std::endl;
	//}
	//system("pause");
	
	//{
	//	measure_time_and_print timer("mat plus:");
	//	vdAdd(Mx*My, x_E_sum, y_E_sum, x_E_sum);
	//	vdAdd(Mx*My, x_E_sum, x_y_mul, x_E_sum);
	//}


	return x_y_mul;
}

int main(int argc, char** argv)
{
    // Set matrix property
    const uint64_t min_col_size = 10;
	vector<uint64_t> row_sizes = {Mx};
	//1000*600 //800*600

    // Measure wall time for each function call
    for (const auto row_size: row_sizes)
    {
    	//const uint64_t col_size = std::max(min_col_size, static_cast<uint64_t>(std::ceil(row_size * 0.6)));
		const uint64_t col_size = static_cast<uint64_t>(std::ceil(row_size * 0.6));
		const uint64_t b_row_size = static_cast<uint64_t>(std::ceil(row_size * 0.8));



    	std::cout << "--- Matrix size are (" << row_size << ", " << col_size << ") and ("<<
    		b_row_size << ", " << col_size << ") ---" << std::endl;

        const MatrixXd a = MatrixXd::Random(row_size, col_size);
        const MatrixXd b = MatrixXd::Random(b_row_size, col_size);

        MatrixXd result_eigen;
        /*{
        measure_time_and_print timer("Eigen function");
        result_eigen = math::euclidean_distance_square_with_eigen(a, b);
        }*/

        const math::matrix a_mat(a);
        const math::matrix b_mat(b);

  //      math::matrix result_loop;
  //      {
  //      measure_time_and_print timer("native function");
  //      result_loop = math::euclidean_distance_square_with_loop(a_mat, b_mat);
  //      }
		//assert(result_loop == result_eigen);

		double* x = (double*)MKL_malloc(Mx*n * sizeof(double), 64);
		double* y = (double*)MKL_malloc(My*n * sizeof(double), 64);
		for (int j = 0; j < n*Mx; j++)
		{
			x[j] = a_mat.data[j];
		}
		for (int i = 0; i < n*My; i++)
		{
			y[i] = b_mat.data[i];
		}
		//colomn domain->row domain
		matrix_cr_conv(x, n, Mx);
		matrix_cr_conv(y, n, My);


		printf("-----------------------------------------\n");

		double *dis;
		{
			measure_time_and_print timer("mkl function");
			dis = MKL_Euclidean_distance(x,y);
		}
			

		//row domain->colomn domain
		matrix_cr_conv(dis, Mx, My);


		//for (int i = 0; i < 20; i++) {
		//	
		//	std::cout << result_eigen.data()[i] << std::endl;
		//	//std::cout << result_loop.data[i] << std::endl;
		//	std::cout << dis[i] << std::endl << std::endl;
		//}

        
		

    }
	system("pause");

    return 0;
}