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

PetscInt layer_num;
Vec *neurons;
Vec *biases;
Mat *weights;
PetscErrorCode ierr;
PetscMPIInt    rank;

void init_network()
{
	layer_num = 3;
	int i;

	int layer_sizes[3] = {28 * 28, 30, 10};
	ierr = PetscMalloc1(layer_num, &neurons);CHKERRQ(ierr);
	for (i = 0; i <layer_num; i++) {
		ierr = VecCreate(PETSC_COMM_WORLD, &neurons[i]);CHKERRQ(ierr);
		ierr = VecSetSizes(neurons[i], PETSC_DECIDE, layer_sizes[i]);CHKERRQ(ierr);
	}

	PetscRandom rnd;
	PetscScalar value;
	PetscRandomCreate(PETSC_COMM_SELF, &rnd);
	ierr = PetscMalloc1(layer_num - 1, &biases);CHKERRQ(ierr);

	int low, high;
	int j;
	for (i = 1; i <layer_num; i++) {
		ierr = VecCreate(PETSC_COMM_WORLD, &biases[i - 1]);CHKERRQ(ierr);
		ierr = VecSetSizes(biases[i - 1], PETSC_DECIDE, layer_sizes[i]);CHKERRQ(ierr);
		
		ierr = VecGetOwnershipRange(biases[i - 1], &low, &high);CHKERRQ(ierr);
	  	for (j=low; j<high; j++) {
	  		ierr = PetscRandomGetValue(rnd, &value);
	    	ierr = VecSetValues(biases[i - 1], 1 , &j, &value, INSERT_VALUES);CHKERRQ(ierr);
	  	}
	  	ierr = VecAssemblyBegin(biases[i - 1]);CHKERRQ(ierr);
  		ierr = VecAssemblyEnd(biases[i - 1]);CHKERRQ(ierr);
	}

	PetscScalar *w;
	ierr = PetscMalloc1(layer_num -1, &w);CHKERRQ(ierr);
	int k, m, n;
	for (i = 1; i <layer_num; i++) {
		ierr = MatCreate(PETSC_COMM_WORLD,&weights[i - 1]);CHKERRQ(ierr);
		ierr = MatSetSizes(weights[i - 1], PETSC_DECIDE, PETSC_DECIDE, m, n);CHKERRQ(ierr);
		ierr = MatSetType(weights[i - 1], MATSEQDENSE);CHKERRQ(ierr);
		m = layer_sizes[i];
		n = layer_sizes[i - 1];
		ierr = PetscMalloc1(m * n, &w);CHKERRQ(ierr); //?? how to set initial value together.
		for (j = 0; j < m; j++) {
			for (k = 0; k < n; k++) {
				ierr = PetscRandomGetValue(rnd, &w[j * m + k]);CHKERRQ(ierr); //? row-wise or column-wise?
			}
		}
		ierr = MatSeqDenseSetPreallocation(weights[i - 1], w);CHKERRQ(ierr);
  		ierr = MatAssemblyBegin(weights[i - 1], MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  		ierr = MatAssemblyEnd(weights[i - 1], MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

		ierr = PetscFree(w);CHKERRQ(ierr);
	}
}

PetscScalar sigmoid(PetscScalar z)
{
    return 1.0/(1.0 + exp(-z));
}

Vec feedforward(Vec x)
{
	int i, j;
	Vec a = x;
	Vec y;
	PetscInt low, high, ni;
	PetscInt *ix;
	PetscScalar *vals;
	for (i = 1; i < layer_num) {
		ierr = MatMultAdd(weights[i - 1], a, biases[i - 1], y);CHKERRQ(ierr);
		ierr = VecGetOwnershipRange(y, &low, &high);CHKERRQ(ierr);
		ni = high - low;

		ierr = PetscMalloc1(ni, &ix);CHKERRQ(ierr);
		for (j = low; j < high; j++) {
			ix[j] = j;
		}
		ierr = VecGetValues(y, ni, ix, vals);CHKERRQ(ierr);
		for (j = 0; j < ni; j++) {
			vals[j] = sigmoid(vals[j]);
		}
	  	ierr = VecSetValues(y, ni, ix, vals, INSERT_VALUES);CHKERRQ(ierr);
		a = y;
	}
	return a;
}

void SGD(Vec *training_data, int iter, int batch_size, float eta)
{
	
}

int main(int argc, char **argv)
{
	PetscInitialize(&argc,&argv);
	
	init_network();


	PetscFinalize();
}