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
PetscInt input_x_size;
PetscInt layer_num;
PetscInt data_size;
PetscInt train_size;
PetscInt test_size;
Vec train_x, train_y;
Vec *neurons;
Vec *biases;
Mat *weights;
Vec *nabla_biases, *delta_nabla_biases;
Mat *nabla_weights, *delta_nabla_weights;
Vec *layer_outputs;
Vec *activations;
PetscErrorCode ierr;
PetscMPIInt    rank;
typedef struct DigitImage
{
	int digit;
	unsigned char *values;
}DigitImage;
DigitImage *train_set;
DigitImage *test_set;

void load_image(char *file_name);
PetscErrorCode init_network();
PetscErrorCode sigmoid(Vec x, Vec r);
PetscErrorCode sigmoid_prime(Vec z, Vec r);
PetscErrorCode feedforward(Vec x, Vec *result);
PetscErrorCode set_vector_x(Vec x, DigitImage *img);
PetscErrorCode set_vector_y(Vec y, DigitImage *img);
PetscErrorCode train_small_batch(int start_pos, int batch_size, float eta);
PetscErrorCode cost_derivative(Vec activation, Vec y);
PetscErrorCode vec2mat_begin(Vec x, PetscScalar **p_arr, Mat *mat);
PetscErrorCode vec2mat_end(Vec x, PetscScalar **p_arr, Mat *mat);
PetscErrorCode backprop();
void shuffle_train_set();
PetscErrorCode create_input_vector();
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
	
	load_image(img_file);

	init_network();

	create_input_vector();

	train(30, 10, 3.0);

	evaluate();

	PetscFinalize();
	
	return 0;	
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

	char *p1, *p2, *p;
	p1 = buf;
	int index = 0;
	char tmp[10];
	int j;

	train_set = (DigitImage *)malloc(train_size * sizeof(DigitImage));
	test_set = (DigitImage *)malloc(test_size * sizeof(DigitImage));

	DigitImage *img_set;
	for (i = 0; i < file_size; i++) {
		if (buf[i] == '\n') {
			if (index < train_size) {
				img_set = train_set;
			} else {
				img_set = test_set;
			}

			p2 = &buf[i];
			strncpy(tmp, p1, 1);

			img_set[index].digit = atof(tmp);
			img_set[index].values = (unsigned char*)malloc(IMAGE_SIZE);
			
			p1 += 2;
			p = p1 + 1;
			j = 0;
			while (p < p2) {
				if (*p == ',' || *p == '\r') {
					strncpy(tmp, p1, p - p1);
					img_set[index].values[j++] = atoi(tmp);
					p1 = p + 1;
					p = p1 + 1;					
				}
				p++;
			}
			index ++;
		}
	}

	free(buf);
	fclose(fp);
}

PetscErrorCode init_network()
{
	layer_num = 3;
	
	int layer_sizes[3] = {IMAGE_SIZE, 30, OUTPUT_SIZE};

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
	return 0;
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
	return 0;
}

PetscErrorCode sigmoid_prime(Vec z, Vec r)
{
	Vec tmp1, tmp2;
	ierr = VecDuplicate(z, &tmp1);CHKERRQ(ierr);
	ierr = sigmoid(z, tmp1);CHKERRQ(ierr);
	ierr = VecDuplicate(tmp1, &tmp2);CHKERRQ(ierr);
	PetscScalar alpha = -1;
	ierr = VecScale(tmp2, alpha);CHKERRQ(ierr);
	ierr = VecPointwiseMult(r, tmp1, tmp2);
	ierr = VecDestroy(&tmp1);CHKERRQ(ierr);
	ierr = VecDestroy(&tmp2);CHKERRQ(ierr);
	return 0;
}

PetscErrorCode feedforward(Vec x, Vec *result)
{
	Vec a;
	Vec y, r;
	PetscInt i;
	ierr = VecDuplicate(x, &a);CHKERRQ(ierr);
	int n = layer_num -1;
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
	return 0;
}

PetscErrorCode set_vector_x(Vec x, DigitImage *img)
{
	PetscInt i, n;
	PetscScalar val;
	
	ierr = VecGetSize(x, &n);CHKERRQ(ierr);
	for (i = 0; i < n; i++) {
		val = img->values[i];
		ierr = VecSetValues(x, 1, &i, &val, INSERT_VALUES);
	}
	VecAssemblyBegin(x);
 	VecAssemblyEnd(x);
 	return 0;
}

PetscErrorCode set_vector_y(Vec y, DigitImage *img)
{
	PetscInt i, n;
	PetscScalar val;
	
	ierr = VecGetSize(y, &n);CHKERRQ(ierr);
 	ierr = VecGetSize(y, &n);CHKERRQ(ierr);
	for (i = 0; i < n; i++) {
		if (i == img->digit) {
			val = 1.0;
		} else {
			val = 0.0;
		}
		ierr = VecSetValues(y, 1, &i, &val, INSERT_VALUES);
	}
	VecAssemblyBegin(y);
 	VecAssemblyEnd(y);
 	return 0;
}


PetscErrorCode train_small_batch(int start_pos, int batch_size, float eta)
{
	int i, j;

	for (i = 0; i < layer_num; i++) {
		ierr = VecZeroEntries(nabla_biases[i]);CHKERRQ(ierr);
		ierr = MatZeroEntries(nabla_weights[i]);CHKERRQ(ierr);
	}
	PetscScalar alpha = 1;
	int n = start_pos + batch_size;
	for (i = start_pos; i < n; i++) {
		set_vector_x(train_x, &train_set[i]);
		set_vector_y(train_y, &train_set[i]);
		backprop();
		for (j = 0; j < layer_num; j++) {
			ierr = VecAXPY(nabla_biases[i], alpha, delta_nabla_biases[i]);CHKERRQ(ierr);
			ierr = MatAXPY(nabla_weights[i], alpha, delta_nabla_weights[i], SAME_NONZERO_PATTERN);CHKERRQ(ierr);
		}

	}

	alpha = - eta/batch_size;
	for (i =0; i < layer_num; i++) {
		ierr = VecAXPY(biases[i], alpha, nabla_biases[i]);CHKERRQ(ierr);
		ierr = MatAXPY(weights[i], alpha, nabla_weights[i], SAME_NONZERO_PATTERN);CHKERRQ(ierr);
	}
	return 0;
}

PetscErrorCode cost_derivative(Vec activation, Vec y)
{
	PetscScalar a = -1;
	ierr = VecAXPY(activation, a, y);CHKERRQ(ierr);
	return 0;
}

PetscErrorCode vec2mat_begin(Vec x, PetscScalar **p_arr, Mat *mat)
{
	PetscScalar *x_arr;
	PetscInt nlocal;
	PetscInt one = 1;
	ierr = VecGetArray(x,	&x_arr);CHKERRQ(ierr);
	p_arr = &x_arr;
	ierr = VecGetLocalSize(x, &nlocal);CHKERRQ(ierr);
	ierr = MatCreateDense(PETSC_COMM_WORLD, nlocal, one, PETSC_DECIDE, PETSC_DECIDE, x_arr, mat);CHKERRQ(ierr);
	return 0;
}

PetscErrorCode vec2mat_end(Vec x, PetscScalar **p_arr, Mat *mat)
{
	ierr = VecRestoreArray(x, p_arr);CHKERRQ(ierr);
	ierr = MatDestroy(mat);CHKERRQ(ierr);
	return 0;
}

PetscErrorCode backprop()
{
	Vec z;	
	Vec activation = train_x;
	Vec *activations;
	Vec *zs;
	PetscInt i, n;
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
	ierr = vec2mat_begin(activations[n - 2], &arr2, &mat_tmp2);CHKERRQ(ierr);
	ierr = MatMatTransposeMult(mat_tmp1, mat_tmp2, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &delta_nabla_weights[n - 1]);CHKERRQ(ierr);
	ierr = vec2mat_end(delta, &arr1, &mat_tmp1);CHKERRQ(ierr);
	ierr = vec2mat_end(activation, &arr2, &mat_tmp2);CHKERRQ(ierr);

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

		ierr = VecCopy(delta, delta_nabla_biases[layer_num - i]);CHKERRQ(ierr);
		
		ierr = vec2mat_begin(delta, &arr1, &mat_tmp1);CHKERRQ(ierr);
		ierr = vec2mat_begin(activations[n - i - 1], &arr2, &mat_tmp2);CHKERRQ(ierr);
		ierr = MatMatTransposeMult(mat_tmp1, mat_tmp2, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &delta_nabla_weights[n - 1]);CHKERRQ(ierr);
		ierr = vec2mat_end(delta, &arr1, &mat_tmp1);CHKERRQ(ierr);
		ierr = vec2mat_end(activations[n - i - 1], &arr2, &mat_tmp2);CHKERRQ(ierr);

		ierr = VecDestroy(&delta);
		ierr = VecDestroy(&delta_tmp);
		ierr = VecDestroy(&sp);
	}
	return 0;
}

void shuffle_train_set()
{
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

    for (i = 0; i < test_size; i++) {
    	set_vector_x(test_x, &test_set[i]);
    	feedforward(test_x, &r);
    	if (rank == 0) {
    		ierr = VecGetValues(r, OUTPUT_SIZE, ix, values);CHKERRQ(ierr);
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
    	printf("\ntest size:%d correct count:%d correct rate:%f\n", test_size, correct_count, correct_rate);
    }
    return 0;
}
