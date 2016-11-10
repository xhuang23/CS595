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
PetscInt layer_num;
PetscInt training_count;
Vec *training_x;
Vec training_y;
Vec *neurons;
Vec *biases;
Mat *weights;
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
			if (training_count  == 10) {
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
	int pixel_count = 28 * 28;
	ierr = PetscMalloc1(training_count, &training_x);CHKERRQ(ierr);
	for (i = 0; i < file_size; i++) {
		if (buf[i] == '\n') {
			ierr = VecCreate(PETSC_COMM_WORLD, &training_x[index]);CHKERRQ(ierr);
			ierr = VecSetSizes(training_x[index], PETSC_DECIDE, pixel_count);CHKERRQ(ierr);
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
}

PetscScalar sigmoid(PetscScalar z)
{
    return 1.0/(1.0 + exp(-z));
}

PetscErrorCode feedforward(Vec x, Vec *result)
{
	Vec a = x;
	Vec y;
	PetscInt i, j, nlocal;
	PetscScalar *array;
	for (i = 1; i < layer_num; i++) {
		ierr = MatMultAdd(weights[i - 1], a, biases[i - 1], y);CHKERRQ(ierr);
		ierr = VecGetLocalSize(y, &nlocal);CHKERRQ(ierr);
 		ierr = VecGetArray(y, &array);CHKERRQ(ierr);
 		for (j = 0; j < nlocal; j++) {
 			array[j] = sigmoid(array[j]);
 		}
 		VecRestoreArray(y, &array);
		a = y;
	}
	*result = a;
}

PetscErrorCode SGD(int iter, int batch_size, float eta)
{
		
}

int main(int argc, char **argv)
{
	char img_file[1024];
	strcpy(img_file, argv[1]);

	PetscInitialize(&argc, &argv, NULL, NULL);
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
	
	load_image(img_file);

	init_network();

	PetscFinalize();
	
}