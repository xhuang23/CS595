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


PetscMPIInt rank;
PetscInt input_x_size;
PetscInt layer_num;
PetscInt training_count;
Vec *training_x;
Vec training_y;

Vec *neurons;
Vec *biases;
Mat *weights;

Vec *nabla_biases, *delta_nabla_biases;
Mat *nabla_weights *delta_nabla_weights;


Vec *layer_outputs;
Vec *activations;
PetscErrorCode ierr;
PetscMPIInt    rank;

PetscErrorCode load_image(char *file_name)
{
	FILE *fp = fopen(file_name, "r");
	size_t file_size;
	fseek(fp, 0L, SEEK_END);
	file_size = ftell(fp);
	rewind(fp);
	char *buf = malloc(file_size);
	fread(buf, 1, file_size, fp);
	int i;
	training_count = 0;
	for (i = 0; i < file_size; i++) {
		if (buf[i] == '\n') {
			training_count++;
			if (training_count == 10) {
				break;
			}
		}
	}

	PetscScalar *x_arr, *y_arr;

	ierr = VecCreate(PETSC_COMM_WORLD, &training_y);CHKERRQ(ierr);
	ierr = VecSetSizes(training_y, PETSC_DECIDE, training_count);CHKERRQ(ierr);
	ierr = VecSetFromOptions(training_y);CHKERRQ(ierr);
	ierr = VecGetArray(training_y, &y_arr);

	char *p1, *p2, *p;
	p1 = buf;
	int index = 0;
	char tmp[10];
	int j;
	int input_x_size = 28 * 28;
	ierr = PetscMalloc1(training_count, &training_x);CHKERRQ(ierr);
	for (i = 0; i < file_size; i++) {
		if (buf[i] == '\n') {
			ierr = VecCreate(PETSC_COMM_WORLD, &training_x[index]);CHKERRQ(ierr);
			ierr = VecSetSizes(training_x[index], PETSC_DECIDE, input_x_size);CHKERRQ(ierr);
			ierr = VecSetFromOptions(training_x[index]);;CHKERRQ(ierr);
			ierr = VecGetArray(training_x[index], &x_arr);

			p2 = &buf[i];
			strncpy(tmp, p1, 1);
			y_arr[index] = atof(tmp);

			p1 += 2;
			p = p1 + 1;
			j = 0;
			while (p < p2) {
				if (*p == ',' || *p == '\r') {
					strncpy(tmp, p1, p - p1);
					x_arr[j++] = atof(tmp);
					p1 = p + 1;
					p = p1 + 1;					
				}
				p++;
			}
			VecRestoreArray(training_x[index], &x_arr);
			index ++;
			if (index == 10)
				break;
		}
	}
	VecRestoreArray(training_y, &y_arr);
	free(buf);
	fclose(fp);
}

PetscErrorCode init_network()
{
	layer_num = 3;
	
	int layer_sizes[3] = {28 * 28, 30, 10};

	PetscRandom rnd;
	PetscScalar value;
	PetscScalar *array;
	PetscRandomCreate(PETSC_COMM_WORLD, &rnd);
	ierr = PetscMalloc1(layer_num - 1, &biases);CHKERRQ(ierr);
	
	ierr = PetscMalloc1(layer_num - 1, &layer_outputs);CHKERRQ(ierr);
	ierr = PetscMalloc1(layer_num, &activations);CHKERRQ(ierr);

	ierr = VecCreate(PETSC_COMM_WORLD, &activations[0]);CHKERRQ(ierr);
	ierr = VecSetSizes(activations[0], PETSC_DECIDE, input_x_size);CHKERRQ(ierr);
	ierr = VecSetFromOptions(activations[0]);CHKERRQ(ierr);

	PetscInt i, j, nlocal;
	for (i = 1; i <layer_num; i++) {
		ierr = VecCreate(PETSC_COMM_WORLD, &biases[i - 1]);CHKERRQ(ierr);
		ierr = VecSetSizes(biases[i - 1], PETSC_DECIDE, layer_sizes[i]);CHKERRQ(ierr);
		ierr = VecSetFromOptions(biases[i - 1]);CHKERRQ(ierr);
		
		ierr = VecGetLocalSize(biases[i - 1], &nlocal);CHKERRQ(ierr);
 		ierr = VecGetArray(biases[i - 1],	&array);CHKERRQ(ierr);
 		for (j = 0; j < nlocal; j++) {
 			ierr = PetscRandomGetValue(rnd, &value);CHKERRQ(ierr);
 			array[j] = value;
 		}
 		VecRestoreArray(biases[i - 1], &array);
 		VecDuplicate(biases[i - 1], &layer_outputs[i - 1]);
 		VecDuplicate(biases[i - 1], &activations[i]);
	}


	PetscInt row, col, m, n;
	ierr = PetscMalloc1(layer_num - 1, &weights);CHKERRQ(ierr);
	for (i = 1; i <layer_num; i++) {
		m = layer_sizes[i];
		n = layer_sizes[i - 1];
		
		ierr = MatCreate(PETSC_COMM_WORLD,&weights[i - 1]);CHKERRQ(ierr);
		ierr = MatSetSizes(weights[i - 1], PETSC_DECIDE, PETSC_DECIDE, m, n);CHKERRQ(ierr);
		ierr = MatSetType(weights[i - 1], MATDENSE);CHKERRQ(ierr);
		ierr = MatSetFromOptions(weights[i - 1]);CHKERRQ(ierr);
		ierr = MatSetUp(weights[i - 1]);CHKERRQ(ierr);

		ierr = MatDenseGetArray(weights[i - 1], &array);CHKERRQ(ierr);
		for (col = 0; col < n; col++) {
		    for (row = 0; row < m; row++) {
		      ierr = PetscRandomGetValue(rnd, &value);CHKERRQ(ierr);
		      array[m * col + row] = value;
		    }
		}
		ierr = MatDenseRestoreArray(weights[i - 1], &array);CHKERRQ(ierr);
	}



	ierr = PetscMalloc1(layer_num - 1, &nabla_biases);CHKERRQ(ierr);
	ierr = PetscMalloc1(layer_num - 1, &delta_nabla_biases);CHKERRQ(ierr);
	for (i = 1; i <layer_num; i++) {
		ierr = VecDuplicate(biases[i - 1], &nabla_biases[i - 1]);CHKERRQ(ierr);
		ierr = VecDuplicate(biases[i - 1], &delta_nabla_biases[i - 1]);CHKERRQ(ierr);
	}

	ierr = PetscMalloc1(layer_num - 1, &nabla_weights);CHKERRQ(ierr);
	ierr = PetscMalloc1(layer_num - 1, &delta_nabla_weights);CHKERRQ(ierr);
	for (i = 1; i <layer_num; i++) {
		ierr = MatDuplicate(weights[i - 1], MAT_DO_NOT_COPY_VALUES, &nabla_weights[i - 1]);
		ierr = MatDuplicate(weights[i - 1], MAT_DO_NOT_COPY_VALUES, &delta_nabla_weights[i - 1]);
	}
}

PetscErrorCode sigmoid(Vec x, Vec r)
{
	PetscInt i, nlocal;
	PetscScalar *array, z;
	
	ierr = VecCopy(x, r);	



	ierr = VecGetLocalSize(r, &nlocal);CHKERRQ(ierr);
	ierr = VecGetArray(r, &array);CHKERRQ(ierr);
	for (i = 0; i < nlocal; i++) {
		z = array[i];
		array[i] = 1.0/(1.0 + exp(-z));
	}
	ierr = VecRestoreArray(r, &array);CHKERRQ(ierr);
}

PetscErrorCode sigmoid_prime(Vec z, Vec r)
{
	Vec tmp1, tmp2;
	ierr = VecDuplicate(z, &tmp1);CHKERRQ(ierr);
	ierr = sigmoid(z, tmp1);CHKERRQ(ierr);
	ierr = VecDuplicate(temp1, &tmp2);CHKERRQ(ierr);
	PetscScalar alpha = -1;
	ierr = VecScale(tmp2, alpha);CHKERRQ(ierr);
	ierr = VecPointwiseMult(r, temp1, temp2);
	ierr = VecDestroy(&tmp1);CHKERRQ(ierr);
	ierr = VecDestroy(&tmp2);CHKERRQ(ierr);
}

PetscErrorCode feedforward(Vec x, Vec *result)
{
	Vec a;
	Vec y, r;
	PetscInt i, j, nlocal;
	PetscScalar *array;
	ierr = VecDuplicate(x, &a);CHKERRQ(ierr);
	int n = layer_num = -1;
	for (i = 0; i < n; i++) {
		ierr = VecDuplicate(biases[i], &y);CHKERRQ(ierr);
		ierr = MatMultAdd(weights[i], a, biases[i], y);CHKERRQ(ierr);
		ierr = VecDuplicate(biases[i], &r);CHKERRQ(ierr);
		sigmoid(y, r);
		ierr = VecDestroy(&y);CHKERRQ(ierr);
		ierr = VecDestroy(&a);CHKERRQ(ierr);
		ierr = VecDuplicate(r, &a);CHKERRQ(ierr);
		ierr = VecDestroy(&r);CHKERRQ(ierr);
	}
	*result = a;
}

PetscErrorCode train_small_batch(Vec *x_batch, Vec *y_batch, int batch_size, float eta)
{
	int i, j;
	for (i = 0; i < layer_num; i++) {
		ierr = VecZeroEntries(nabla_biases[i]);CHKERRQ(ierr);
		ierr = MatZeroEntries(nabla_weights[i]);CHKERRQ(ierr);
	}
	PetscScalar alpha = 1;
	for (i = 0; i < batch_size; i++) {
		backprop(x_batch[i], y_batch[i]);
		for (j = 0; j < layer_num; j++) {
			ierr = VecAXPY(nabla_biases[i], alpha, delta_nabla_biases[i]);CHKERRQ(ierr);
			ierr = MatAXPY(nabla_weights[i], alpha, delta_nabla_weights[i]);CHKERRQ(ierr);
		}
	}

	alpha = - eta/batch_size;
	for (i =0; i < layer_num; i++) {
		ierr = VecAXPY(biases[i], alpha, nabla_biases[i]);CHKERRQ(ierr);
		ierr = MatAXPY(weights[i], alpha, nabla_weights[i]);CHKERRQ(ierr);
	}
}

PetscErrorCode cost_derivative(Vec activation, Vec y)
{
	PetscScalar a = -1;
	ierr = VecAXPY(activation, a, y);CHKERRQ(ierr);
}

PetscErrorCode vec2mat_begin(Vec x, PetscScalar **p_arr, Mat *mat) {
	PetscScalar *x_arr;
	PetscScalar nlocal;
	PetscInt one = 1;
	ierr = VecGetArray(x,	&x_arr);CHKERRQ(ierr);
	p_arr = &x_arr;
	ierr = VecGetLocalSize(x, &nlocal);CHKERRQ(ierr);
	ierr = MatCreateDense(PETSC_COMM_WORLD, nlocal, one, PETSC_DECIDE, PETSC_DECIDE, x_arr, mat);CHKERRQ(ierr);
}

PetscErrorCode vec2mat_end(Vec x, PetscScalar **p_arr, Mat *mat) {
	ierr = VecRestoreArray(x, p_arr);CHKERRQ(ierr);
	ierr = MatDestroy(mat);CHKERRQ(ierr);
}

PetscErrorCode backprop(Vec x, Vec y)
{
	Vec z;	
	Vec activation = x;
	Vec *activations;
	Vec *zs;
	PetscInt i, j, n, nlocal;
	PetscScalar *array;
	n = layer_num - 1;

	for (i = 0; i < n; i++) {
		ierr = VecZeroEntries(delta_nabla_biases[i]);CHKERRQ(ierr);
		ierr = MatZeroEntries(delta_nabla_weights[i]);CHKERRQ(ierr);
	}

	ierr = PetscMalloc1(n, &activations);CHKERRQ(ierr);
	ierr = PetscMalloc1(n, &zs);CHKERRQ(ierr);
	for (i = 0; i < n; i++) {
		ierr = MatMultAdd(weights[i], activation, biases[i], layer_outputs[i]);CHKERRQ(ierr);
		sigmoid(layer_outputs[i], activations[i]);
	}

	Vec tmp_vec1;
	ierr = VecDuplicate(activations[n - 1], &tmp_vec1);CHKERRQ(ierr);
	ierr = cost_derivative(tmp_vec1, y);CHKERRQ(ierr);
	Vec tmp_vec2;
	ierr = VecDuplicate(layer_outputs[n - 1], &tmp_vec2);CHKERRQ(ierr);
	sigmoid_prime(layer_outputs[n - 1], tmp_vec2);CHKERRQ(ierr);

	Vec delta;
	ierr = VecDuplicate(tmp_vec1, &delta);CHKERRQ(ierr);
	ierr = VecPointwiseMult(delta, temp1, temp2);CHKERRQ(ierr);
	
	ierr = VecDestroy(&tmp_vec1);CHKERRQ(ierr);
	ierr = VecDestroy(&tmp_vec2);CHKERRQ(ierr);

	ierr = VecCopy(delta, delta_nabla_biases[n - 1]);CHKERRQ(ierr);

	Mat mat_tmp1, mat_tmp2;
	PetscScalar **p_arr1;
	PetscScalar **p_arr2;
	ierr = vec2mat_begin(delta, p_arr1, &mat_tmp1);CHKERRQ(ierr);
	ierr = vec2mat_begin(activations[n - 2];, p_arr2, &mat_tmp2);CHKERRQ(ierr);
	ierr = MatMatTransposeMult(mat_tmp1, mat_tmp2, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &delta_nabla_weights[n - 1]);CHKERRQ(ierr);
	ierr = vec2mat_end(delta, p_arr1, &mat_tmp1);CHKERRQ(ierr);
	ierr = vec2mat_end(activation, p_arr2, &mat_tmp2);CHKERRQ(ierr);

	Vec sp;
	Mat mat_tmp;
	Vec delta_tmp;

	for (i = 2; i < layer_num; i++) {
		z = layer_outputs[layer_num - i];
		
		ierr = VecDuplicate(z, &sp);CHKERRQ(ierr);
		sigmoid_prime(z, sp);
		ierr = MatCreateTranspose(weights[layer_num - i + 1], &mat_tmp);CHKERRQ(ierr);
		ierr = VecDuplicate(delta, &delta_tmp);CHKERRQ(ierr);
		ierr = MatMult(mat_tmp, delta, delta_tmp);
		ierr = VecPointwiseMult(delta, delta_tmp, sp);CHKERRQ(ierr);

		ierr = VecCopy(delta, delta_nabla_b[layer_num - i]);CHKERRQ(ierr);
		
		ierr = vec2mat_begin(delta, p_arr1, &mat_tmp1);CHKERRQ(ierr);
		ierr = vec2mat_begin(activations[n - i - 1], p_arr2, &mat_tmp2);CHKERRQ(ierr);
		ierr = MatMatTransposeMult(mat_tmp1, mat_tmp2, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &delta_nabla_weights[n - 1]);CHKERRQ(ierr);
		ierr = vec2mat_end(delta, p_arr1, &mat_tmp1);CHKERRQ(ierr);
		ierr = vec2mat_end(activations[n - i - 1], p_arr2, &mat_tmp2);CHKERRQ(ierr);

		ierr = VecDestroy(&delta);
		ierr = VecDestroy(&delta_tmp);
		ierr = VecDestroy(&sp);
	}
}

PetscErrorCode SGD(int iter, int batch_size, float eta)
{
	int i, k;
	for (i = 0; i < iter; i++) {
		//shuffle training data
		for (k = 0; k < training_count; k += batch_size) {
			train_small_batch(x_batch, y_batch, batch_size, eta);
		}
	}
}

int main(int argc, char **argv)
{
	if (argc < 2) {
		printf("Usgae ./digit <training_file>\n");
		return;
	}
	char img_file[1024];
	strcpy(img_file, argv[1]);

	PetscInitialize(&argc, &argv, NULL, NULL);
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
	
	load_image(img_file);

	init_network();

	//SGD(30, 10, )

	PetscFinalize();
	
}