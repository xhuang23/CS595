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

Vec *nabla_biases;
Mat *nabla_weights;

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

	ierr = PetscMalloc1(layer_num - 1, &nabla_biases);CHKERRQ(ierr);
	for (i = 1; i <layer_num; i++) {
		ierr = VecDuplicate(biases[i - 1], &nabla_biases[i - 1]);CHKERRQ(ierr);
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

	ierr = PetscMalloc1(layer_num - 1, &nabla_weights);CHKERRQ(ierr);
	for (i = 1; i <layer_num; i++) {
		
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
	VecRestoreArray(r, &array);
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
	Vec a = x;
	Vec y;
	PetscInt i, j, nlocal;
	PetscScalar *array;
	for (i = 1; i < layer_num; i++) {
		ierr = MatMultAdd(weights[i - 1], a, biases[i - 1], y);CHKERRQ(ierr);
		sigmoid(y);
		a = y;
	}
	*result = a;
}

PetscErrorCode train_small_batch(Vec *x_batch, Vec y_batch, int batch_size, float eta)
{
	/*
	nabla_b = [np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]
    for x, y in mini_batch:
        delta_nabla_b, delta_nabla_w = self.backprop(x, y)
        nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
    self.weights = [w-(eta/len(mini_batch))*nw
                    for w, nw in zip(self.weights, nabla_w)]
    self.biases = [b-(eta/len(mini_batch))*nb
                   for b, nb in zip(self.biases, nabla_b)]
	*/


}

PetscErrorCode cost_derivative(Vec activation, Vec y)
{
	PetscScalar a = -1;
	ierr = VecAXPY(activation, a, y);CHKERRQ(ierr);
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

}



PetscErrorCode get_training_data()

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

	SGD(30, 10, )

	PetscFinalize();
	
}