#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <time.h>
#include <string.h>
#include <petscsys.h>
#include <petscvec.h>
#include <petscmat.h>

#define IMAGE_SIZE 28*28
#define OUTPUT_SIZE 10

PetscMPIInt rank;
PetscInt layer_num;
PetscInt data_size;
PetscInt train_size;
PetscInt test_size;
Mat train_x, train_y;
Mat *neurons;
Mat *biases;
Mat *weights;
Mat *nabla_biases, *delta_nabla_biases;
Mat *nabla_weights, *delta_nabla_weights;
Mat *layer_outputs;
Mat *activations;
PetscErrorCode ierr;
PetscMPIInt    rank;
typedef struct DigitImage
{
	int digit;
	unsigned char *values;
}DigitImage;
DigitImage *train_set;
DigitImage *test_set;

void parse_image(char *buf, size_t buf_size, size_t *p_start, DigitImage *img_set, int set_size);
void load_image(char *file_name);
PetscErrorCode init_network();
PetscErrorCode sigmoid(Mat x, Mat r);
PetscErrorCode sigmoid_prime(Mat z, Mat r);
PetscErrorCode feedforward(Mat x, Mat *result);
PetscErrorCode set_input_x(Mat x, DigitImage *img);
PetscErrorCode set_input_y(Mat y, DigitImage *img);
PetscErrorCode train_small_batch(int start_pos, int batch_size, float eta);
PetscErrorCode cost_derivative(Mat activation, Mat y);
PetscErrorCode backprop();
void shuffle_train_set();
PetscErrorCode create_input_mat();
PetscErrorCode train(int iter, int batch_size, float eta);
PetscErrorCode evaluate();

int main(int argc, char **argv)
{
	if (argc < 2) {
		printf("Usgae ./digit <training_file>\n");
		return 0;
	}
	time_t t;
	srand((unsigned) time(&t));

	char img_file[1024];
	strcpy(img_file, argv[1]);

	PetscInitialize(&argc, &argv, NULL, NULL);
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
	
	char buffer[100] = {'\0'};
	if (rank == 0) {
		load_image(img_file);
		sprintf(buffer, "%d,%d", train_size, test_size);	
	}

	MPI_Bcast(buffer, 100, MPI_CHAR, 0, MPI_COMM_WORLD);
	
	if (rank != 0) {
		const char sep[2] = ",";
		char *token;
		token = strtok(buffer, sep);
		train_size = atoi(token);
		token = strtok(NULL, sep);
		test_size = atoi(token);
	}

	init_network();

	create_input_vector();

	train(30, 10, 3.0);

	evaluate();

	PetscFinalize();
	
	return 0;	
}

void parse_image(char *buf, size_t buf_size, size_t *p_start, DigitImage *img_set, int set_size)
{
	char *p1, *p2, *p;
	int index = 0;
	char tmp[10];
	size_t i;
	int j;
	int start_pos = *p_start;
	p1 = buf + start_pos;
	for (i = start_pos; i < buf_size; i++) {
		if (buf[i] == '\n') {
			p2 = &buf[i];

			tmp[0] = *p1;
			tmp[1] = '\0';

			img_set[index].digit = atoi(tmp);
			img_set[index].values = (unsigned char*)malloc(IMAGE_SIZE);
			
			p1 += 2;
			p = p1 + 1;
			j = 0;
			while (p < p2) {
				if (*p == ',' || *p == '\r') {
					strncpy(tmp, p1, p - p1);
					tmp[p - p1] = '\0';
					img_set[index].values[j++] = atoi(tmp);
					p1 = p + 1;
					if (*p == '\r') {
						p1 += 1;	
					}
					p = p1 + 1;					
				} else {
					p++;	
				}
			}
			index ++;
			if (index == set_size) {
				break;
			}
		}
	}
	*p_start = i + 1;
}

void load_image(char *file_name)
{
	FILE *fp = fopen(file_name, "r");
	size_t file_size;
	fseek(fp, 0L, SEEK_END);
	file_size = ftell(fp);
	rewind(fp);
	char *buf = malloc(file_size);
	fread(buf, 1, file_size, fp);
	size_t i;
	data_size = 0;
	for (i = 0; i < file_size; i++) {
		if (buf[i] == '\n') {
			data_size++;
		}
	}

	train_size = data_size * 2 / 3;
	test_size = data_size -  train_size;
	train_set = (DigitImage *)malloc(train_size * sizeof(DigitImage));
	test_set = (DigitImage *)malloc(test_size * sizeof(DigitImage));
	
	size_t start_pos = 0;
	parse_image(buf, file_size, &start_pos, train_set, train_size);
	parse_image(buf, file_size, &start_pos, test_set, test_size);

	free(buf);
	fclose(fp);
}

PetscErrorCode create_dense_mat(PetscInt m, PetscInt n, Mat *A)
{
	ierr = MatCreate(PETSC_COMM_WORLD, A);CHKERRQ(ierr);
	ierr = MatSetSizes(*A, PETSC_DECIDE, PETSC_DECIDE, m, n);CHKERRQ(ierr);
	ierr = MatSetType(*A, MATELEMENTAL);CHKERRQ(ierr);
	ierr = MatSetUp(*A);CHKERRQ(ierr);	
	return 0;
}

PetscErrorCode init_mat_random(Mat A, PetscRandom rnd)
{
	IS isrows, iscols;
	PetscInt *rows, *cols, nrows, ncols;
	PetscInt i, j;
	PetscScalar *arr, rval;

	ierr = MatGetOwnershipIS(A, &isrows, &iscols);CHKERRQ(ierr);
	ierr = ISGetLocalSize(isrows, &nrows);CHKERRQ(ierr);
	ierr = ISGetIndices(isrows, &rows);CHKERRQ(ierr);
	ierr = ISGetLocalSize(iscols, &ncols);CHKERRQ(ierr);
	ierr = ISGetIndices(iscols, &cols);CHKERRQ(ierr);

	ierr = PetscMalloc1(nrows * ncols, &arr);CHKERRQ(ierr);
	for (i = 0; i < nrows; i++) {
		for (j = 0; j < ncols; j++) {
			ierr = PetscRandomGetValue(rnd, &rval);CHKERRQ(ierr);
	  		arr[i * ncols + j] = rval;
	}

	ierr = MatSetValues(A, nrows, rows, ncols, cols, arr, INSERT_VALUES);CHKERRQ(ierr);
	ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	ierr = ISRestoreIndices(isrows, &rows);CHKERRQ(ierr);
	ierr = ISRestoreIndices(iscols, &cols);CHKERRQ(ierr);
	ierr = ISDestroy(&isrows);CHKERRQ(ierr);
	ierr = ISDestroy(&iscols);CHKERRQ(ierr);

	return 0;
}

PetscErrorCode init_network()
{
	layer_num = 3;
	int layer_sizes[3] = {IMAGE_SIZE, 30, OUTPUT_SIZE};

	PetscRandom rnd;
	PetscInt one = 1;
	PetscRandomCreate(PETSC_COMM_WORLD, &rnd);

	ierr = PetscMalloc1(layer_num - 1, &biases);CHKERRQ(ierr);
	ierr = PetscMalloc1(layer_num - 1, &layer_outputs);CHKERRQ(ierr);
	ierr = PetscMalloc1(layer_num, &activations);CHKERRQ(ierr);

	create_dense_mat(IMAGE_SIZE, one, &activations[0]);

	PetscInt i, j, nlocal;
	for (i = 1; i <layer_num; i++) {
		create_dense_mat(layer_sizes[i], one, &biases[i - 1]);
		init_mat_random(biases[i - 1], rnd);
 		ierr = MatDuplicate(biases[i - 1], MAT_DO_NOT_COPY_VALUES, &layer_outputs[i - 1]);
 		ierr = MatDuplicate(biases[i - 1], MAT_DO_NOT_COPY_VALUES, &activations[i]);
	}

	PetscInt row, col, m, n, mlocal;
	ierr = PetscMalloc1(layer_num - 1, &weights);CHKERRQ(ierr);
	for (i = 1; i <layer_num; i++) {
		m = layer_sizes[i];
		n = layer_sizes[i - 1];
		create_dense_mat(m, n, &weights[i - 1]);
		init_mat_random(weights[i - 1], rnd);
	}

	ierr = PetscMalloc1(layer_num - 1, &nabla_biases);CHKERRQ(ierr);
	ierr = PetscMalloc1(layer_num - 1, &delta_nabla_biases);CHKERRQ(ierr);
	for (i = 1; i <layer_num; i++) {
		ierr = MatDuplicate(biases[i - 1], MAT_DO_NOT_COPY_VALUES, &nabla_biases[i - 1]);
		ierr = MatDuplicate(biases[i - 1], MAT_DO_NOT_COPY_VALUES, &delta_nabla_biases[i - 1]);
	}

	ierr = PetscMalloc1(layer_num - 1, &nabla_weights);CHKERRQ(ierr);
	ierr = PetscMalloc1(layer_num - 1, &delta_nabla_weights);CHKERRQ(ierr);
	for (i = 1; i <layer_num; i++) {
		ierr = MatDuplicate(weights[i - 1], MAT_DO_NOT_COPY_VALUES, &nabla_weights[i - 1]);
		ierr = MatDuplicate(weights[i - 1], MAT_DO_NOT_COPY_VALUES, &delta_nabla_weights[i - 1]);
	}
	return 0;
}

PetscErrorCode sigmoid(Mat A, Mat B)
{
	IS isrows, iscols;
	PetscInt *rows, *cols, nrows, ncols;
	PetscInt i, j, k;
	PetscScalar *arr, rval, temp;;

	ierr = MatCopy(A, B, SAME_NONZERO_PATTERN);CHKERRQ(ierr);
	ierr = MatGetOwnershipIS(B, &isrows, &iscols);CHKERRQ(ierr);
	ierr = ISGetLocalSize(isrows, &nrows);CHKERRQ(ierr);
	ierr = ISGetIndices(isrows, &rows);CHKERRQ(ierr);
	ierr = ISGetLocalSize(iscols, &ncols);CHKERRQ(ierr);
	ierr = ISGetIndices(iscols, &cols);CHKERRQ(ierr);

	ierr = PetscMalloc1(nrows * ncols, &arr);CHKERRQ(ierr);	
	ierr = MatSetValues(B, nrows, rows, ncols, cols, arr);CHKERRQ(ierr);
	for (i = 0; i < nrows; i++) {
		for (j = 0; j < ncols; j++) {
			k = i * ncols + j;
			temp = 1.0/(1.0 + exp(-arr[k]));
			arr[k] = temp;
	}
	ierr = MatSetValues(B, nrows, rows, ncols, cols, arr, INSERT_VALUES);CHKERRQ(ierr);
	ierr = MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	ierr = MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	ierr = ISRestoreIndices(isrows, &rows);CHKERRQ(ierr);
	ierr = ISRestoreIndices(iscols, &cols);CHKERRQ(ierr);
	ierr = ISDestroy(&isrows);CHKERRQ(ierr);
	ierr = ISDestroy(&iscols);CHKERRQ(ierr);
	return 0;
}

PetscErrorCode sigmoid_prime(Mat A, Mat B)
{
	IS isrows, iscols;
	PetscInt *rows, *cols, nrows, ncols;
	PetscInt i, j, k;
	PetscScalar *arr, rval, temp;

	ierr = MatCopy(A, B, SAME_NONZERO_PATTERN);CHKERRQ(ierr);
	ierr = MatGetOwnershipIS(B, &isrows, &iscols);CHKERRQ(ierr);
	ierr = ISGetLocalSize(isrows, &nrows);CHKERRQ(ierr);
	ierr = ISGetIndices(isrows, &rows);CHKERRQ(ierr);
	ierr = ISGetLocalSize(iscols, &ncols);CHKERRQ(ierr);
	ierr = ISGetIndices(iscols, &cols);CHKERRQ(ierr);

	ierr = PetscMalloc1(nrows * ncols, &arr);CHKERRQ(ierr);	
	ierr = MatSetValues(B, nrows, rows, ncols, cols, arr);CHKERRQ(ierr);
	for (i = 0; i < nrows; i++) {
		for (j = 0; j < ncols; j++) {
			k = i * ncols + j;
			temp = 1.0/(1.0 + exp(-arr[k]));
			arr[k] = temp * (1.0 - temp);
	}
	ierr = MatSetValues(B, nrows, rows, ncols, cols, arr, INSERT_VALUES);CHKERRQ(ierr);
	ierr = MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	ierr = MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	ierr = ISRestoreIndices(isrows, &rows);CHKERRQ(ierr);
	ierr = ISRestoreIndices(iscols, &cols);CHKERRQ(ierr);
	ierr = ISDestroy(&isrows);CHKERRQ(ierr);
	ierr = ISDestroy(&iscols);CHKERRQ(ierr);
	return 0;
}

PetscErrorCode feedforward(Mat A, Mat *B)
{
	int i;
	int n = layer_num -1;
	
	Mat *temp_activations;
	ierr = PetscMalloc1(layer_num, &temp_activations);CHKERRQ(ierr);
	for (i = 0; i < layer_num; i++) {
		ierr = MatDuplicate(activations[i], MAT_DO_NOT_COPY_VALUES, &temp_activations[i]);
	}

	ierr = PetscMalloc1(layer_num, &temp_outputs);CHKERRQ(ierr);
	for (i = 0; i < n; i++) {
		ierr = MatDuplicate(biases[i], MAT_DO_NOT_COPY_VALUES, &temp_outputs[i]);
	}	

	ierr = MatCopy(A, temp_activations[0], SAME_NONZERO_PATTERN);CHKERRQ(ierr);
	for (i = 0; i < n; i++) {
		ierr = MatMultAdd(weights[i], temp_activations1[i], biases[i], temp_outputs[i]);CHKERRQ(ierr);
		sigmoid(temp_outputs[i], temp_activations[i + 1]);
	}

	ierr = MatDuplicate(temp_activations[layer_num - 1], MAT_COPY_VALUES, R);

	for (i = 0; i < layer_num; i++) {
		ierr = MatDestroy(&temp_activations[i]);CHKERRQ(ierr);
	}

	for (i = 0; i < n; i++) {
		ierr = MatDestroy(&temp_outputs[i]);CHKERRQ(ierr);
	}

	ierr = PetscFree(temp_activations);CHKERRQ(ierr);
	ierr = PetscFree(temp_outputs);CHKERRQ(ierr);

	return 0;
}

PetscErrorCode set_input_x(MAT A, DigitImage *img)
{
	PetscInt *rows, *cols, nrows, ncols;
	PetscInt i, j, k;
	PetscScalar *arr, val;

	if (rank == 0) {
		nrows = IMAGE_SIZE;
		ncols = 1;
		ierr = PetscMalloc1(nrows, rows);
		ierr = PetscMalloc1(ncols, cols);
		for (i = 0; i < nrows; i++) {
			rows[i] = i;
		}
		for (i = 0; i < ncols; i++) {
			cols[i] = i;
		}
		ierr = PetscMalloc1(nrows * ncols, &arr);CHKERRQ(ierr);
		for (i = 0; i < nrows; i++) {
			for (j = 0; j < ncols; j++) {
				k = i * ncols + j;
		  		arr[k] = img->values[k];
		}
		ierr = MatSetValues(A, nrows, rows, ncols, cols, arr, INSERT_VALUES);CHKERRQ(ierr);
		
	}
	ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	if (rank == 0) {
		ierr = PetscFree(rows);CHKERRQ(ierr);
		ierr = PetscFree(cols);CHKERRQ(ierr);
		ierr = PetscFree(arr);CHKERRQ(ierr);
	}
 	return 0;
}

PetscErrorCode set_input_y(Mat A, DigitImage *img)
{
	PetscInt *rows, *cols, nrows, ncols;
	PetscInt i, j, k;
	PetscScalar *arr, val;

	if (rank == 0) {
		nrows = OUTPUT_SIZE;
		ncols = 1;
		ierr = PetscMalloc1(nrows, rows);
		ierr = PetscMalloc1(ncols, cols);
		for (i = 0; i < nrows; i++) {
			rows[i] = i;
		}
		for (i = 0; i < ncols; i++) {
			cols[i] = i;
		}
		ierr = PetscMalloc1(nrows * ncols, &arr);CHKERRQ(ierr);
		for (i = 0; i < nrows; i++) {
			for (j = 0; j < ncols; j++) {
				k = i * ncols + j;
		  		arr[k] = img->digit;
		}
		ierr = MatSetValues(A, nrows, rows, ncols, cols, arr, INSERT_VALUES);CHKERRQ(ierr);
		
	}
	ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	if (rank == 0) {
		ierr = PetscFree(rows);CHKERRQ(ierr);
		ierr = PetscFree(cols);CHKERRQ(ierr);
		ierr = PetscFree(arr);CHKERRQ(ierr);
	}
 	return 0;
}


PetscErrorCode train_small_batch(int start_pos, int batch_size, float eta)
{
	int i, j;
	int m = layer_num - 1;
	
	for (i = 0; i < m; i++) {
		ierr = MatZeroEntries(nabla_biases[i]);CHKERRQ(ierr);
		ierr = MatZeroEntries(nabla_weights[i]);CHKERRQ(ierr);
	}

	PetscScalar alpha = 1;
	int n = start_pos + batch_size;
	for (i = start_pos; i < n; i++) {

		set_input_x(train_x, &train_set[i]);
		set_input_y(train_y, &train_set[i]);

		backprop();
		
		for (j = 0; j < m; j++) {
			ierr = MatAXPY(nabla_biases[j], alpha, delta_nabla_biases[j], SAME_NONZERO_PATTERN);CHKERRQ(ierr);
			ierr = MatAXPY(nabla_weights[j], alpha, delta_nabla_weights[j], SAME_NONZERO_PATTERN);CHKERRQ(ierr);
		}
	}

	alpha = - eta/batch_size;
	for (j =0; j < m; j++) {
		ierr = MatAXPY(biases[j], alpha, nabla_biases[j], SAME_NONZERO_PATTERN);CHKERRQ(ierr);
		ierr = MatAXPY(weights[j], alpha, nabla_weights[j], SAME_NONZERO_PATTERN);CHKERRQ(ierr);
	}
	return 0;
}

PetscErrorCode cost_derivative(Mat A, Mat B)
{
	PetscScalar alpha = -1;
	ierr = MatAXPY(A, alpha, B);CHKERRQ(ierr);
	return 0;
}

PetscErrorCode backprop()
{
	Vec z;
	ierr = MatCopy(train_x, activations[0], SAME_NONZERO_PATTERN);CHKERRQ(ierr);
	PetscInt i, n;
	PetscScalar alpha = 1.0;
	n = layer_num - 1;

	for (i = 0; i < n; i++) {
		ierr = MatZeroEntries(delta_nabla_biases[i]);CHKERRQ(ierr);
		ierr = MatZeroEntries(delta_nabla_weights[i]);CHKERRQ(ierr);
	}

	for (i = 0; i < n; i++) {
		ierr = MatMatMult(weights[i], activations[i], MAT_REUSE_MATRIX, PETSC_DEFAULT, &layer_outputs[i]);CHKERRQ(ierr);
		ierr = MatAXPY(layer_outputs[i], alpha, biases[i]);CHKERRQ(ierr);
		sigmoid(layer_outputs[i], activations[i + 1]);
	}

	Vec tmp_vec1;
	ierr = VecDuplicate(activations[layer_num - 1], &tmp_vec1);CHKERRQ(ierr);
	ierr = cost_derivative(tmp_vec1, train_y);CHKERRQ(ierr);
	Vec tmp_vec2;
	ierr = VecDuplicate(layer_outputs[n - 1], &tmp_vec2);CHKERRQ(ierr);
	sigmoid_prime(layer_outputs[n - 1], tmp_vec2);CHKERRQ(ierr);

	Vec delta;
	ierr = VecDuplicate(tmp_vec1, &delta);CHKERRQ(ierr);
	ierr = VecPointwiseMult(delta, tmp_vec1, tmp_vec2);CHKERRQ(ierr);
	
	ierr = VecDestroy(&tmp_vec1);CHKERRQ(ierr);
	ierr = VecDestroy(&tmp_vec2);CHKERRQ(ierr);

	ierr = VecCopy(delta, delta_nabla_biases[n - 1]);CHKERRQ(ierr);

	Mat mat_tmp1, mat_tmp2;
	PetscScalar *arr1;
	PetscScalar *arr2;
	ierr = vec2mat_begin(delta, &arr1, &mat_tmp1);CHKERRQ(ierr);
	ierr = vec2mat_begin(activations[layer_num - 2], &arr2, &mat_tmp2);CHKERRQ(ierr);
	
	Mat mat_tmp2_transpose;
	ierr = MatTranspose(mat_tmp2, MAT_INITIAL_MATRIX, &mat_tmp2_transpose);CHKERRQ(ierr);
	ierr = MatMatMult(mat_tmp1 ,mat_tmp2_transpose, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &delta_nabla_weights[n - 1]);CHKERRQ(ierr);
	ierr = MatDestroy(&mat_tmp2_transpose);CHKERRQ(ierr);
	
	ierr = vec2mat_end(delta, &arr1, &mat_tmp1);CHKERRQ(ierr);
	ierr = vec2mat_end(activations[n - 2], &arr2, &mat_tmp2);CHKERRQ(ierr);

	Vec sp;
	Mat mat_tmp;
	
	for (i = 2; i < layer_num; i++) {
		z = layer_outputs[n - i];
		ierr = VecDuplicate(z, &sp);CHKERRQ(ierr);
		sigmoid_prime(z, sp);

		ierr = MatCreateTranspose(weights[n - i + 1], &mat_tmp);CHKERRQ(ierr);
		ierr = MatMult(mat_tmp, delta_nabla_biases[n - i + 1], delta_nabla_biases[n - i]);
		ierr = MatDestroy(&mat_tmp);CHKERRQ(ierr);

		ierr = VecPointwiseMult(delta_nabla_biases[n - i], delta_nabla_biases[n - i], sp);CHKERRQ(ierr);

		ierr = vec2mat_begin(delta_nabla_biases[n - i], &arr1, &mat_tmp1);CHKERRQ(ierr);
		ierr = vec2mat_begin(activations[layer_num - i - 1], &arr2, &mat_tmp2);CHKERRQ(ierr);
		
		ierr = MatMatTransposeMult(mat_tmp1, mat_tmp2, MAT_REUSE_MATRIX, PETSC_DEFAULT, &delta_nabla_weights[n - i]);
		
		ierr = vec2mat_end(delta_nabla_biases[n - i], &arr1, &mat_tmp1);CHKERRQ(ierr);
		ierr = vec2mat_end(activations[layer_num - i - 1], &arr2, &mat_tmp2);CHKERRQ(ierr);

		ierr = VecDestroy(&sp);
	}
	return 0;
}

void shuffle_train_set()
{
	if (rank != 0)
		return;

	int i, pos, digit;
	unsigned char *p;
	for (i = 0; i < train_size; i++) {
		pos = rand() % train_size;
		if (pos != i) {
			digit = train_set[pos].digit;
			p = train_set[pos].values;
			train_set[pos].digit = train_set[i].digit;
			train_set[pos].values = train_set[i].values;
			train_set[i].digit = digit;
			train_set[i].values = p;
		}
	}
}

PetscErrorCode create_input_vector()
{
	ierr = VecCreate(PETSC_COMM_WORLD, &train_x);CHKERRQ(ierr);
	ierr = VecSetSizes(train_x, PETSC_DECIDE, IMAGE_SIZE);CHKERRQ(ierr);
	ierr = VecSetFromOptions(train_x);CHKERRQ(ierr);

	ierr = VecCreate(PETSC_COMM_WORLD, &train_y);CHKERRQ(ierr);
	ierr = VecSetSizes(train_y, PETSC_DECIDE, OUTPUT_SIZE);CHKERRQ(ierr);
	ierr = VecSetFromOptions(train_y);CHKERRQ(ierr);
	return 0;
}

PetscErrorCode train(int iter, int batch_size, float eta)
{
	int i, k;
	int real_batch_size;

	for (i = 0; i < iter; i++) {
		shuffle_train_set();
		for (k = 0; k < train_size; k += batch_size) {
			if (k + batch_size - 1 < train_size) {
				real_batch_size = batch_size;
			} else {
				real_batch_size = train_size - k;
			}

			train_small_batch(k, real_batch_size, eta);
		}
		printf("Iteration %d\n", i + 1);
	}
	return 0;
}

PetscErrorCode evaluate()
{
	int i, j, max_pos;
    Vec r;
	Vec test_x;
    PetscInt ix[OUTPUT_SIZE];
    PetscScalar values[OUTPUT_SIZE];
    PetscScalar max_value;

    PetscInt correct_count = 0;

    if (rank == 0) {
    	for (i = 0; i < OUTPUT_SIZE; i++) {
    		ix[i] = i;	
    	}
    }

    ierr = VecDuplicate(train_x, &test_x);CHKERRQ(ierr);

    char str_tmp[1024] = {'\0'};
    for (i = 0; i < test_size; i++) {
    	set_vector_x(test_x, &test_set[i]);	

    	feedforward(test_x, &r);
    	if (rank == 0) {
    		ierr = VecGetValues(r, OUTPUT_SIZE, ix, values);CHKERRQ(ierr);
    		sprintf(str_tmp,"%f , %f , %f, %f , %f , %f, %f , %f , %f, %f\n", values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7], values[8], values[9]);
    		printf(str_tmp);
    		max_value = values[0];
    		max_pos = 0;
    		for (j = 0; j < OUTPUT_SIZE; j++) {
    			if (values[j] > max_value) {
    				max_value = values[j];
    				max_pos = j; 
    			}
    		}
    		if (max_pos == test_set[i].digit) {
    			correct_count++;
    		}
    	}
    	ierr = VecDestroy(&r);CHKERRQ(ierr);
    }

    if (rank == 0) {
    	float correct_rate = correct_count * 1.0/ test_size;
    	printf("test size:%d correct count:%d correct rate:%f\n", test_size, correct_count, correct_rate);
    }
    return 0;
}
