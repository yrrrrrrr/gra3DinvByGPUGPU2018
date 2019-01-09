/*****************************************************************************
Copyright: 2019, Yuan Yongqi.
File name: kernel.cu
Description: The parallel calculation of cpu and gpu for multi-faceted gravity
anomaly forward, the bott method for inversion of three-dimensional positive
triangle mesh interface and the nonlinear method are implemented, and the
time consumption under the forward motion of cpu and gpu is compared.
Author: Yuan Yongqi
Version: 1.3
Date: 2019.1.2
*****************************************************************************/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <stdarg.h>

#define G 6.67e-11*1e5*1000*1000
#define PI       3.1415926
/*Relatively constant density*/
#define P      1

/*setting for GPU forward8*/
#define BLOCK_SIZE 128
/*Forward index data width*/
#define NCOLUMN 12

/*Matrix structure*/
typedef struct {
	/* DIM */
	int rows;
	int cols;
	float** data;
} Matrix;

/*Basic operation macro definition*/
#define NORM(a,b,c) sqrtf(pow((a),2) + pow((b),2) + pow((c),2))
#define DOT(a1,a2,a3,b1,b2,b3) (a1)*(b1) + (a2)*(b2) + (a3)*(b3)
#define CROSSX(a1,a2,a3,b1,b2,b3) ((a2)*(b3)-(a3)*(b2))/ \
	(sqrtf(pow((a2)*(b3) - (a3)*(b2), 2) + \
	pow(a3*b1 - a1*b3, 2) + pow(a1*b2 - b1*a2, 2)))
#define CROSSY(a1,a2,a3,b1,b2,b3) ((a3)*(b1)-(a1)*(b3))/ \
	(sqrtf(pow((a2)*(b3) - (a3)*(b2), 2) + \
	pow((a3)*(b1) - (a1)*(b3), 2) + pow((a1)*(b2) - (b1)*(a2), 2)))
#define CROSSZ(a1,a2,a3,b1,b2,b3) ((a1)*(b2)-(b1)*(a2))/ \
	(sqrtf(pow((a2)*(b3) - (a3)*(b2), 2) + \
	pow((a3)*(b1) - (a1)*(b3), 2) + pow((a1)*(b2) - (b1)*(a2), 2)))
#define SIGN(a) ((a)<0)?-1.0:(((a)<=0.000001)?0.0:1.0)

/*************************************************
Function: ReadData
Description: Read three columns of data in a file
if bFlag=1
input:const char *chFileName,
	bool bFlag, int *iNum,
	int iRow, float *aX, float *aY, float *aZ
output: The number of rows in the matrix in the file
if bFlag=0
input:const char *chFileName,
	bool bFlag, int *iNum,
	int iRow, float *aX, float *aY, float *aZ
output: Read file data is written to three arrays
*************************************************/
__host__ void ReadData(const char *chFileName,
	bool bFlag, int *iNum,
	int iRow, float *aX, float *aY, float *aZ);

/*************************************************
Function: Offset
Description : Offset the observation point(x,y)and make z=0
input : int iNum, float *aX, float *aY, float *aZ
output : float *aX, float *aY, float *aZ
*************************************************/
__host__ void Offset(int iNum, float *aX, float *aY, float *aZ);

/*************************************************
Function: MaxDiff
Description : Find the absolute value of the maximum error of two vectors
input :  float A[], float B[], int iLength
output : return
*************************************************/
__host__ float MaxDiff(float A[], float B[], int iLength);

/*************************************************
Function: getGPUInfo
Description : Get the parameter information of gpu
input : NULL
output : Print in console
*************************************************/
__host__ void getGPUInfo();

/*************************************************
Function: forwardIndex
Description : Generate triangle-specific data for
forward modeling based on tri and data
input :float *aTriX, float *aTriY, float *aTriZ, int iTriNum,
	float *aDataX, float *aDataY, float *aDataZ
output : float A[][NCOLUMN], float B[][NCOLUMN],
example:
tri:
1  4  5
point:
(datax,datay,dataz)
matrixA
point1(X,Y,Z),point4(X,Y,Z),point5(X,Y,Z),point1(X,Y,Z)
matrixB
point1(X,Y,0),point4(X,Y,0),point5(X,Y,0),point1(X,Y,0)
*************************************************/
__host__ void forwardIndex(float A[][NCOLUMN], float B[][NCOLUMN],
	float *aTriX, float *aTriY, float *aTriZ, int iTriNum,
	float *aDataX, float *aDataY, float *aDataZ);

/*************************************************
Function: lineMethod
Description :*Holstein, H., Ketteridge, B., 1996.
Gravimetric analysis of uniform polyhedra. Geophysics 61, 357–364.
*Holstein, H., Schürholz, P., Starr, A. J. and Chakraborty, M., 1999,
*Comparison of gravimetric formulas for uniform
polyhedra: Geophysics, 64, 1438–1446.
input :aObsX,aObsY,aObsZ,index[][12],star,end
output : return
*************************************************/
__host__ float lineMethod(float aObsX, float aObsY, float aObsZ,
	float index[][12], int star, int end);

/*************************************************
Function: normPrint
Description :Print the two norms of the difference between the two vectors
input :float arrA[], float arrB[], int N
output : Print in console
*************************************************/
__host__ void normPrint(float arrA[], float arrB[], int N);

/* Convert to unit matrix */
__host__ void set_identity_matrix(Matrix m);

/* Allocate initial space for the matrix */
Matrix alloc_matrix(int rows, int cols);

/* Exchange two rows of the  matrix */
__host__ void swap_rows(Matrix m, int r1, int r2);

/* Multiply a row of a matrix by a factor */
__host__ void scale_row(Matrix m, int r, float scalar);

/* Add scalar * row r2 to row r1. */
__host__ void shear_row(Matrix m, int r1, int r2, float scalar);

/* Uses Gauss-Jordan elimination.
Inversion of the matrix (learn from others)
The elimination procedure works by applying elementary row
operations to our input matrix until the input matrix is reduced to
the identity matrix.
Simultaneously, we apply the same elementary row operations to a
separate identity matrix to produce the inverse matrix.
If this makes no sense, read wikipedia on Gauss-Jordan elimination.

This is not the fastest way to invert matrices, so this is quite
possibly the bottleneck. */
__host__ int destructive_invert_matrix(Matrix input, Matrix output);

/*************************************************
Function: JacobiIndex
Description : Generate the index used to calculate the Jacobian matrix
input :float *aTriX, float *aTriY, float *aTriZ, int iTriNum,
	float *aDataX, float *aDataY, float *aDataZ,
	int maxFaceNum, int iDataNum, 
output : float A[][NCOLUMN], float B[][NCOLUMN],*aOffset
*************************************************/
__host__ void JacobiIndex(
	float *aTriX, float *aTriY, float *aTriZ, int iTriNum,
	float *aDataX, float *aDataY, float *aDataZ,
	int maxFaceNum, int iDataNum, int *aOffset);

/*************************************************
Function:JacobiCPU
Description :Cpu's Jacobian matrix calculation
input :float *aDataX, float *aDataY, float *aDataZ,
	float *aTriX, float *aTriY, float *aTriZ,
	float *aObsX, float *aObsY, float *aObsZ,
	int iTriNum, int iDataNum, int iObsNum,
output : float *result
*************************************************/
__host__ void JacobiCPU(float *aDataX, float *aDataY, float *aDataZ,
	float *aTriX, float *aTriY, float *aTriZ,
	float *aObsX, float *aObsY, float *aObsZ,
	int iTriNum, int iDataNum, int iObsNum,
	float *result);

/*************************************************
Function:getSolve
Description :Complete the iterative quantity based on
the Tikhonov regularization
input : int iDataNum, int iObsNum, float lambda,
	float *J, float *aObsG, float *aDataZ, 
output : float *aDataZ, 
*************************************************/
__host__ void getSolve(int iDataNum, int iObsNum, float lambda,
	float *J, float *aObsG, float *aDataZ, float *aForwardG);

/*************************************************
Function:getTopG
Description :
input :const char *chFileName,
	int iObsNum, float aObsX[], float aObsY[], float aObsZ[],
output :float *out
*************************************************/
__host__ void getTopG(const char *chFileName,
	int iObsNum, float aObsX[], float aObsY[], float aObsZ[],
	float *out);
//device function

/*************************************************
Function:  lineMethod
Description :Gpu device function
input :float obs_x, float obs_y,
float obs_z, float *index, int j
output : return
*************************************************/
__device__ float lineMethod(float obs_x, float obs_y,
	float obs_z, float *index, int j);

/*************************************************
Function:  Kernelreduce
Description :Gpu device function
input :const float *a
output : float *r
*************************************************/
__global__ void Kernelreduce(const float *a, float *r);

/*************************************************
Function:  KernelTop
Description :Gpu device function
input :float  float *d_obsX, float *d_obsY, float *d_obsZ,
	float *d_Arraytop, float *d_Arraybot,
	const int triNum, const int PointPerThreads
output : *midResult,
*************************************************/
__global__ void KernelTop(float *midResult, float *d_obsX, float *d_obsY, float *d_obsZ,
	float *d_Arraytop, 
	const int triNum, const int PointPerThreads);


//forward

__host__ void forwardGravityCPU(float *aDataX, float *aDataY, float *aDataZ,
	float *aTriX, float *aTriY, float *aTriZ,
	float *aObsX, float *aObsY, float *aObsZ,
	int iTriNum, int iObsNum, float aTopG[],
	float *out);

void forwardGravityGPU(float *aDataX, float *aDataY, float *aDataZ,
	float *aTriX, float *aTriY, float *aTriZ,
	float *aObsX, float *aObsY, float *aObsZ,
	int iTriNum, int iObsNum, float topG[],
	float *Forward);

//inversion


/*************************************************
Function:BottCPU
Description :Cpu bott method interface inversion
input : float *aDataX, float *aDataY, float *aObsG,
float *aTriX, float *aTriY, float *aTriZ,
int iTriNum, int iDataNum,
int iterate,
output : float *aForwardG, float *invZ
*************************************************/
__host__ void BottCPU(float *aDataX, float *aDataY, float *aObsG,
	float *aTriX, float *aTriY, float *aTriZ,
	int iTriNum, int iDataNum,
	int iterate, float *aTopG, float *aForwardG, float *invZ, bool bFlag);

/*************************************************
Function:BottGPU
Description :Gpu bott method interface inversion
input : float *aDataX, float *aDataY, float *aObsG,
float *aTriX, float *aTriY, float *aTriZ,
int iTriNum, int iDataNum,
int iterate,
output : float *aForwardG, float *invZ
*************************************************/
__host__ void BottGPU(float *aDataX, float *aDataY, float *aObsG,
	float *aTriX, float *aTriY, float *aTriZ,
	int iTriNum, int iDataNum, 
	int iterate, float *aTopG, float *aForwardG, float *invZ, bool bFlag);

/*************************************************
Function:NolineCPU
Description :Nonlinear interface inversion of cpu
input :float *aDataX, float *aDataY,
float *aTriX, float *aTriY, float *aTriZ,
float *aObsX, float *aObsY, float *aObsZ, float *aObsG,
int iTriNum, int iDataNum, int iObsNum,
int iterate, float intiValue, float lambda,float *aTopG
output :float *invZ,float *aForwardG
*************************************************/
__host__ void NolineCPU(float *aDataX, float *aDataY, float *invZ,
	float *aTriX, float *aTriY, float *aTriZ,
	float *aObsX, float *aObsY, float *aObsZ, float *aObsG,
	int iTriNum, int iDataNum, int iObsNum,
	int iterate, float intiValue, float lambda, float *aTopG,
	float *aForwardG, bool bFlag);

/*************************************************
Function:NolineGPU
Description :Nonlinear interface inversion of gpu
input :float *aDataX, float *aDataY,
float *aTriX, float *aTriY, float *aTriZ,
float *aObsX, float *aObsY, float *aObsZ, float *aObsG,
int iTriNum, int iDataNum, int iObsNum,
int iterate, float intiValue, float lambda
output :float *invZ,float *aForwardG
*************************************************/
__host__ void NolineGPU(float *aDataX, float *aDataY, float *invZ,
	float *aTriX, float *aTriY, float *aTriZ,
	float *aObsX, float *aObsY, float *aObsZ, float *aObsG,
	int iTriNum, int iDataNum, int iObsNum,
	int iterate, float intiValue, float lambda, float *aTopG,
	float *aForwardG, bool bFlag);


int main()
{

	//char *chDataFile = "data54_71.txt";
	//char *chTriFile = "tri54_71.txt";
	//char *chObsFile = "obs54_71.txt";
	//char *chBlnFile = "bln54_71.txt";

	//char *chDataFile = "data47_61.txt";
	//char *chTriFile = "tri47_61.txt";
	//char *chObsFile = "obs47_61.txt";
	//char *chBlnFile = "bln47_61.txt";

	//char *chDataFile = "data39_51.txt";
	//char *chTriFile = "tri39_51.txt";
	//char *chObsFile = "obs39_51.txt";
	//char *chBlnFile = "bln39_51.txt";

	//char *chDataFile = "data31_41.txt";
	//char *chTriFile = "tri31_41.txt";
	//char *chObsFile = "obs31_41.txt";
	//char *chBlnFile = "bln31_41.txt";

	//char *chDataFile = "data24_31.txt";
	//char *chTriFile = "tri24_31.txt";
	//char *chObsFile = "obs24_31.txt";
	//char *chBlnFile = "bln24_31.txt";

	//char *chDataFile = "data16_21.txt";
	//char *chTriFile = "tri16_21.txt";
	//char *chObsFile = "obs16_21.txt";
	//char *chBlnFile = "bln16_21.txt";

	//char *chDataFile = "data8_11.txt";
	//char *chTriFile = "tri8_11.txt";
	//char *chObsFile = "obs8_11.txt";
	//char *chBlnFile = "bln8_11.txt";

	char *chDataFile = "data4_6.txt";
	char *chTriFile = "tri4_6.txt";
	char *chObsFile = "obs4_6.txt";
	char *chBlnFile = "bln4_6.txt";

	int iDataNum, iTriNum, iObsNum;//原始数据行数//索引行数//索引行数

	ReadData(chDataFile, true, &iDataNum, 0, NULL, NULL, NULL);
	printf("iDataNum:  %d\n", iDataNum);
	float *aDataX = (float*)malloc(sizeof(float)*iDataNum);//测点与模型相同
	float *aDataY = (float*)malloc(sizeof(float)*iDataNum);//测点与模型相同在类bott法中
	float *aDataZ = (float*)malloc(sizeof(float)*iDataNum);
	ReadData(chDataFile, false, NULL, iDataNum, aDataX, aDataY, aDataZ);

	ReadData(chTriFile, true, &iTriNum, 0, NULL, NULL, NULL);
	printf("iTriNum:   %d\n", iTriNum);
	float *aTriX = (float*)malloc(sizeof(float)*iTriNum);
	float *aTriY = (float*)malloc(sizeof(float)*iTriNum);
	float *aTriZ = (float*)malloc(sizeof(float)*iTriNum);
	ReadData(chTriFile, false, NULL, iTriNum, aTriX, aTriY, aTriZ);

	ReadData(chObsFile, true, &iObsNum, 0, NULL, NULL, NULL);
	printf("iObsNum:   %d\n", iObsNum);
	float *aObsX = (float*)malloc(sizeof(float)*iObsNum);
	float *aObsY = (float*)malloc(sizeof(float)*iObsNum);
	float *aObsZ = (float*)malloc(sizeof(float)*iObsNum);
	float *aObsG = (float*)malloc(sizeof(float)*iObsNum);
	ReadData(chObsFile, false, NULL, iObsNum, aObsX, aObsY, aObsG);
	Offset(iObsNum, aObsX, aObsY, aObsZ);
	

	//getGPUInfo();

	float *aTopG = (float*)malloc(sizeof(float)*iObsNum);
	getTopG(chBlnFile, iObsNum, aObsX, aObsY, aObsZ, aTopG);

	clock_t start, finish;
	double totalTime = 0.0;

	////CPU and GPU forward
	//printf("\n     (CPU and GPU forward)\n");

	//float *aForCPU = (float*)malloc(sizeof(float)*iObsNum);
	//float *aForGPU = (float*)malloc(sizeof(float)*iObsNum);

	//printf("\n A,Cpu running\n");
	//start = clock();

	//forwardGravityCPU(aDataX, aDataY, aDataZ,
	//	aTriX, aTriY, aTriZ,
	//	aObsX, aObsY, aObsZ,
	//	iTriNum, iObsNum, aTopG,
	//	aForCPU);

	//finish = clock();
	//totalTime = (double)(finish - start) / CLOCKS_PER_SEC;
	//printf("\nThe total time is %lf seconds!\n", totalTime);

	//printf("\n B,GPU running\n");
	//start = clock();

	//forwardGravityGPU(aDataX, aDataY, aDataZ,
	//	aTriX, aTriY, aTriZ,
	//	aObsX, aObsY, aObsZ,
	//	iTriNum, iObsNum, aTopG,
	//	aForGPU);

	//finish = clock();
	//totalTime = (double)(finish - start) / CLOCKS_PER_SEC;
	//printf("\nThe total time is %lf seconds!\n", totalTime);

	//FILE *fp;
	//fp = fopen("out8_11g.txt", "w+");
	//for (int i = 0; i<iObsNum; i++)
	//	fprintf(fp, "%f  %f   %f\n ", aObsX[i],aObsY[i],aForGPU[i]);
	//fclose(fp);
	//FILE *fp1;
	//fp1 = fopen("out8_11c.txt", "w+");
	//for (int i = 0; i<iObsNum; i++)
	//	fprintf(fp1, "%f  %f   %f\n ", aObsX[i], aObsY[i], aForGPU[i]);
	//fclose(fp1);


	float lambda = 1;
	int iterate = 1;
	float intiValue = 1;
	bool show = 0;


	////Bott inversion
	//printf("\n     (Bott inversion runningtime)\n");
	//float *aInvGCPU = (float*)malloc(sizeof(float)*iObsNum);
	//float *aInvGGPU = (float*)malloc(sizeof(float)*iObsNum);
	//float *aInvZCPU = (float*)malloc(sizeof(float)*iDataNum);
	//float *aInvZGPU = (float*)malloc(sizeof(float)*iDataNum);
	//printf("\n A,CPU running\n");
	//start = clock();
	//BottCPU(aDataX, aDataY, aObsG,
	//	aTriX, aTriY, aTriZ,
	//	iTriNum, iDataNum,
	//	iterate, aTopG, aInvGCPU, aInvZCPU,show);
	//finish = clock();
	//totalTime = (double)(finish - start) / CLOCKS_PER_SEC;
	//printf("\nThe total time is %lf seconds!\n", totalTime);
	//printf("\n B,GPU running\n");
	//start = clock();
	//BottGPU(aDataX, aDataY, aObsG,
	//	aTriX, aTriY, aTriZ,
	//	iTriNum, iDataNum,
	//	iterate, aTopG, aInvGGPU, aInvZGPU, show);
	//finish = clock();
	//totalTime = (double)(finish - start) / CLOCKS_PER_SEC;
	//printf("\nThe total time is %lf seconds!\n", totalTime);




	//Nonlinear inversion
	printf("\n     (Noline inversion runningtime)\n");
	float *aNolineGCPU = (float*)malloc(sizeof(float)*iObsNum);
	float *aNolineGGPU = (float*)malloc(sizeof(float)*iObsNum);
	float *aNolineZCPU = (float*)malloc(sizeof(float)*iDataNum);
	float *aNolineZGPU = (float*)malloc(sizeof(float)*iDataNum);
	printf("\n A,Cpu running\n");
	start = clock();
	NolineCPU(aDataX, aDataY, aNolineZCPU,
		aTriX, aTriY, aTriZ,
		aObsX, aObsY, aObsZ, aObsG,
		iTriNum, iDataNum, iObsNum,
		iterate, intiValue, lambda, aTopG, aNolineGCPU, show);
	finish = clock();
	totalTime = (double)(finish - start) / CLOCKS_PER_SEC;
	printf("\nThe total time is %lf seconds!\n", totalTime);
	printf("\n B,Gpu running\n");
	start = clock();
	NolineGPU(aDataX, aDataY, aNolineZGPU,
		aTriX, aTriY, aTriZ,
		aObsX, aObsY, aObsZ, aObsG,
		iTriNum, iDataNum, iObsNum,
		iterate, intiValue, lambda, aTopG, aNolineGGPU, show);
	finish = clock();
	totalTime = (double)(finish - start) / CLOCKS_PER_SEC;
	printf("\nThe total time is %lf seconds!\n", totalTime);


	return 0;
}

__host__ void ReadData(const char *chFileName,
	bool bFlag, int *iNum,
	int iRow, float *aX, float *aY, float *aZ)
{
	FILE *fp;
	fp = fopen(chFileName, "r");
	if (fp == NULL)
	{
		printf("fail to open the file！\n");
	}
	if (bFlag == true)
	{
		int Count = 0;
		float Temp;
		while (fscanf(fp, "%f", &Temp) != EOF)
		{
			Count++;
		}
		*iNum = Count / 3;
	}
	else
	{
		for (int i = 0; i < iRow; i++)
		{
			fscanf(fp, "%f %f %f", &aX[i], &aY[i], &aZ[i]);
		}
	}
	fclose(fp);
}

__host__ void Offset(int iNum, float *aX, float *aY, float *aZ)
{
	for (int i = 0; i < iNum; i++)
	{
		aX[i] = aX[i] + 0.1;
		aY[i] = aY[i] + 0.1;
		aZ[i] = 0;
	}
}

__host__ float MaxDiff(float A[], float B[], int iLength)
{
	float fMaxDiff = 0;
	for (int i = 0; i < iLength; i++)
	{
		if (fMaxDiff < abs(A[i] - B[i]))
		{
			fMaxDiff = abs(A[i] - B[i]);
		}
	}
	return fMaxDiff;
}

__host__ void forwardIndex(float A[][NCOLUMN], 
	float *aTriX, float *aTriY, float *aTriZ, int iTriNum,
	float *aDataX, float *aDataY, float *aDataZ)//下界面
{
	for (int i = 0; i < iTriNum; i++)
	{
		int  a = aTriX[i] - 1;//index1to3
		int  b = aTriY[i] - 1;//index4to6
		int  c = aTriZ[i] - 1;//index7to9

		A[i][0] = aDataX[a];
		A[i][1] = aDataY[a];
		A[i][2] = aDataZ[a];

		A[i][3] = aDataX[b];
		A[i][4] = aDataY[b];
		A[i][5] = aDataZ[b];

		A[i][6] = aDataX[c];
		A[i][7] = aDataY[c];
		A[i][8] = aDataZ[c];

		A[i][9] = aDataX[a];
		A[i][10] = aDataY[a];
		A[i][11] = aDataZ[a];
	}
}

__host__ float lineMethod(float aObsX, float aObsY, float aObsZ,
	float index[][12], int star, int end)
{
	/*
	*Holstein, H., Ketteridge, B., 1996. Gravimetric analysis of uniform polyhedra. Geophysics 61, 357–364.
	*Holstein, H., Schürholz, P., Starr, A. J. and Chakraborty, M., 1999,
	*Comparison of gravimetric formulas for uniform polyhedra: Geophysics, 64, 1438–1446.
	*input:obs(x,y,z),index,row of index
	*output:value for the uniform polyherdra
	*/
	float CalcGravity = 0;
	for (int j = star; j < end; j++)
	{
		float a1 = index[j][3] - index[j][0],
			a2 = index[j][4] - index[j][1],
			a3 = index[j][5] - index[j][2];
		float b1 = index[j][6] - index[j][3],
			b2 = index[j][7] - index[j][4],
			b3 = index[j][8] - index[j][5];

		float dnx = CROSSX(a1, a2, a3, b1, b2, b3);
		float dny = CROSSY(a1, a2, a3, b1, b2, b3);
		float dnz = CROSSZ(a1, a2, a3, b1, b2, b3);

		float lineGravity = 0;
		for (int i = 0; i < 3; i++)
		{
			float a1 = index[j][i * 3],
				a2 = index[j][i * 3 + 1],
				a3 = index[j][i * 3 + 2];
			float b1 = index[j][i * 3 + 3],
				b2 = index[j][i * 3 + 4],
				b3 = index[j][i * 3 + 5];

			float L = NORM(a1 - b1, a2 - b2, a3 - b3);

			float dtx = (b1 - a1) / L;
			float dty = (b2 - a2) / L;
			float dtz = (b3 - a3) / L;

			float dhx = CROSSX(dtx, dty, dtz, dnx, dny, dnz);
			float dhy = CROSSY(dtx, dty, dtz, dnx, dny, dnz);
			float dhz = CROSSZ(dtx, dty, dtz, dnx, dny, dnz);

			float vr1x = a1 - aObsX;
			float vr1y = a2 - aObsY;
			float vr1z = a3 - aObsZ;

			float vr2x = b1 - aObsX;
			float vr2y = b2 - aObsY;
			float vr2z = b3 - aObsZ;

			float v = DOT(dnx, dny, dnz, vr1x, vr1y, vr1z);
			float h = DOT(dhx, dhy, dhz, vr2x, vr2y, vr2z);
			float r1 = NORM(vr1x, vr1y, vr1z);
			float r2 = NORM(vr2x, vr2y, vr2z);

			float DN = L / (r2 + r1);

			float C = 2 * atanh(DN);
			float E = 0.5*(r1 + r2 - L*DN);
			float Q = (SIGN(v)) * 2 * atan(h*DN / (abs(v) + E));

			lineGravity = lineGravity + dnz*(h*C - v*Q);
		}
		CalcGravity = CalcGravity + lineGravity;
	}
	return CalcGravity;
}

__device__ float lineMethod(float obs_x, float obs_y,
	float obs_z, float* index, int j)
{
	float a1 = index[j * 12 + 3] - index[j * 12 + 0],
		a2 = index[j * 12 + 4] - index[j * 12 + 1],
		a3 = index[j * 12 + 5] - index[j * 12 + 2];
	float b1 = index[j * 12 + 6] - index[j * 12 + 3],
		b2 = index[j * 12 + 7] - index[j * 12 + 4],
		b3 = index[j * 12 + 8] - index[j * 12 + 5];

	float dnx = CROSSX((double)a1, (double)a2, (double)a3, 
		(double)b1, (double)b2, (double)b3);
	float dny = CROSSY((double)a1, (double)a2, (double)a3,
		(double)b1, (double)b2, (double)b3);
	float dnz = CROSSZ((double)a1, (double)a2, (double)a3,
		(double)b1, (double)b2, (double)b3);

	float LineGravity = 0;
	for (int i = 0; i < 3; i++)//线积分
	{
		float a1 = index[j * 12 + i * 3], a2 = index[j * 12 + i * 3 + 1], a3 = index[j * 12 + i * 3 + 2];
		float b1 = index[j * 12 + i * 3 + 3], b2 = index[j * 12 + i * 3 + 4], b3 = index[j * 12 + i * 3 + 5];


		float L = NORM((double)a1 - (double)b1, (double)a2 - (double)b2, (double)a3 - (double)b3);

		//printf("L: %f\n", L);

		float dtx = ((double)b1 - (double)a1) / (double)L;
		float dty = ((double)b2 - (double)a2) / (double)L;
		float dtz = ((double)b3 - (double)a3) / (double)L;

		float dhx = CROSSX((double)dtx, (double)dty, (double)dtz, 
			(double)dnx, (double)dny, (double)dnz);
		float dhy = CROSSY((double)dtx, (double)dty, (double)dtz,
			(double)dnx, (double)dny, (double)dnz);
		float dhz = CROSSZ((double)dtx, (double)dty, (double)dtz,
			(double)dnx, (double)dny, (double)dnz);

		float vr1x = (double)a1 - (double)obs_x;
		float vr1y = (double)a2 - (double)obs_y;
		float vr1z = (double)a3 - (double)obs_z;

		float vr2x = (double)b1 - (double)obs_x;
		float vr2y = (double)b2 - (double)obs_y;
		float vr2z = (double)b3 - (double)obs_z;

		float v = DOT((double)dnx, (double)dny, (double)dnz,
			(double)vr1x, (double)vr1y, (double)vr1z);
		float h = DOT((double)dhx, (double)dhy, (double)dhz,
			(double)vr2x, (double)vr2y, (double)vr2z);
		float r1 = NORM((double)vr1x, (double)vr1y, (double)vr1z);
		float r2 = NORM((double)vr2x, (double)vr2y, (double)vr2z);


		float DN = (double)L / ((double)r2 + (double)r1);
		//printf("DN: %f\n", DN);
		float C = (double)2 * (double)atanh((double)DN);
		//printf("C: %f\n",C);

		float E = (double)0.5*((double)r1 + (double)r2 - (double)L*(double)DN);
		float Q = (double)(SIGN(v)) * (double)2.0 * atan((double)h*(double)DN / (abs((double)v) + (double)E));

		LineGravity = LineGravity + (double)dnz*((double)h*(double)C - (double)v*(double)Q);
		//printf("LineGravity: %f\n", LineGravity);
	}

	return LineGravity;
}


__host__ void normPrint(float arrA[], float arrB[], int N)
{
	float sumNorm = 0;
	for (int i = 0; i < N; i++)
	{
		sumNorm += pow((arrA[i] - arrB[i]), 2);
	}
	printf("\n norm:   %f", sqrt(sumNorm));
}

__host__ void set_identity_matrix(Matrix m) {
	int i;
	int j;
	assert(m.rows == m.cols);
	for (i = 0; i < m.rows; ++i) {
		for (j = 0; j < m.cols; ++j) {
			if (i == j) {
				m.data[i][j] = 1.0;
			}
			else {
				m.data[i][j] = 0.0;
			}
		}
	}
}

Matrix alloc_matrix(int rows, int cols) {
	Matrix m;
	int i;
	int j;
	m.rows = rows;
	m.cols = cols;
	m.data = (float**)malloc(sizeof(float*)* m.rows);

	for (i = 0; i < m.rows; ++i)
	{
		m.data[i] = (float*)malloc(sizeof(float)* m.cols);
		assert(m.data[i]);
		for (j = 0; j < m.cols; ++j) {
			m.data[i][j] = 0.0;
		}
	}
	return m;
}

__host__ void swap_rows(Matrix m, int r1, int r2) {
	float *tmp;
	assert(r1 != r2);
	tmp = m.data[r1];
	m.data[r1] = m.data[r2];
	m.data[r2] = tmp;
}

__host__ void scale_row(Matrix m, int r, float scalar) {
	int i;
	assert(scalar != 0.0);
	for (i = 0; i < m.cols; ++i) {
		m.data[r][i] *= scalar;
	}
}

__host__ void shear_row(Matrix m, int r1, int r2, float scalar) {
	int i;
	assert(r1 != r2);
	for (i = 0; i < m.cols; ++i) {
		m.data[r1][i] += scalar * m.data[r2][i];
	}
}

__host__ int destructive_invert_matrix(Matrix input, Matrix output) {
	int i;
	int j;
	int r;
	float scalar;
	float shear_needed;
	assert(input.rows == input.cols);
	assert(input.rows == output.rows);
	assert(input.rows == output.cols);

	set_identity_matrix(output);

	/* Convert input to the identity matrix via elementary row operations.
	The ith pass through this loop turns the element at i,i to a 1
	and turns all other elements in column i to a 0. */

	for (i = 0; i < input.rows; ++i) {

		if (input.data[i][i] == 0.0) {
			/* We must swap rows to get a nonzero diagonal element. */

			for (r = i + 1; r < input.rows; ++r) {
				if (input.data[r][i] != 0.0) {
					break;
				}
			}
			if (r == input.rows) {
				/* Every remaining element in this column is zero, so this
				matrix cannot be inverted. */
				return 0;
			}
			swap_rows(input, i, r);
			swap_rows(output, i, r);
		}

		/* Scale this row to ensure a 1 along the diagonal.
		We might need to worry about overflow from a huge scalar here. */
		scalar = 1.0 / input.data[i][i];
		scale_row(input, i, scalar);
		scale_row(output, i, scalar);

		/* Zero out the other elements in this column. */
		for (j = 0; j < input.rows; ++j) {
			if (i == j) {
				continue;
			}
			shear_needed = -input.data[j][i];
			shear_row(input, j, i, shear_needed);
			shear_row(output, j, i, shear_needed);
		}
	}

	return 1;
}

__host__ void JacobiIndex(float A[][NCOLUMN], float B[][NCOLUMN],
	float *aTriX, float *aTriY, float *aTriZ, int iTriNum,
	float *aDataX, float *aDataY, float *aDataZ, int maxFaceNum, int iDataNum, int *aOffset)//,int *end
{
	float *flagA = (float*)malloc(sizeof(float)*iDataNum*iTriNum);
	int *J = (int*)malloc(sizeof(int)*maxFaceNum*iDataNum);
	for (int i = 0; i < iDataNum; i++)
	{
		for (int j = 0; j < iTriNum; j++)
		{
			flagA[i*iTriNum + j] = 0;
			if (aTriX[j] - 1 == i)
				flagA[i*iTriNum + j] = 1;
			if (aTriY[j] - 1 == i)
				flagA[i*iTriNum + j] = 1;
			if (aTriZ[j] - 1 == i)
				flagA[i*iTriNum + j] = 1;
		}
	}

	for (int i = 0; i < iDataNum; i++)
	{
		for (int j = 0; j < 6; j++)
		{
			J[i*maxFaceNum + j] = -1;
		}
	}

	for (int i = 0; i < iDataNum; i++)
	{
		int k = 0;
		for (int j = 0; j < iTriNum; j++)
		{
			if (flagA[i*iTriNum + j] == 1)
			{
				J[i*maxFaceNum + k] = j;
				//printf("%d\t", J[i*maxFaceNum + k]);
				k++;
			}
		}
		//array[i] = k;
		//printf("\n");
	}

	for (int k = 0; k < iDataNum; k++)
	{
		for (int i = 0; i < maxFaceNum; i++)
		{
			for (int j = 0; j < 12; j++)
			{
				A[i + k*maxFaceNum][j] = 0;
				B[i + k*maxFaceNum][j] = 0;
			}
		}
	}

	for (int n = 0; n < iDataNum; n++)
	{
		int face = 0;
		for (int i = 0; i < maxFaceNum; i++)
		{
			int k = J[i + n*maxFaceNum];
			if (k >= 0)
			{
				face++;
				int  a = aTriX[k] - 1;//index1to3
				int  b = aTriY[k] - 1;//index4to6
				int  c = aTriZ[k] - 1;//index7to9

				B[i + n*maxFaceNum][0] = aDataX[a];
				B[i + n*maxFaceNum][1] = aDataY[a];
				B[i + n*maxFaceNum][2] = aDataZ[a];
				B[i + n*maxFaceNum][3] = aDataX[b];
				B[i + n*maxFaceNum][4] = aDataY[b];
				B[i + n*maxFaceNum][5] = aDataZ[b];
				B[i + n*maxFaceNum][6] = aDataX[c];
				B[i + n*maxFaceNum][7] = aDataY[c];
				B[i + n*maxFaceNum][8] = aDataZ[c];
				B[i + n*maxFaceNum][9] = aDataX[a];
				B[i + n*maxFaceNum][10] = aDataY[a];
				B[i + n*maxFaceNum][11] = aDataZ[a];

				aDataZ[n] = aDataZ[n] + 1;//增量

				A[i + n*maxFaceNum][0] = aDataX[a];
				A[i + n*maxFaceNum][1] = aDataY[a];
				A[i + n*maxFaceNum][2] = aDataZ[a];
				A[i + n*maxFaceNum][3] = aDataX[b];
				A[i + n*maxFaceNum][4] = aDataY[b];
				A[i + n*maxFaceNum][5] = aDataZ[b];//增量
				A[i + n*maxFaceNum][6] = aDataX[c];
				A[i + n*maxFaceNum][7] = aDataY[c];
				A[i + n*maxFaceNum][8] = aDataZ[c];
				A[i + n*maxFaceNum][9] = aDataX[a];
				A[i + n*maxFaceNum][10] = aDataY[a];
				A[i + n*maxFaceNum][11] = aDataZ[a];

				aDataZ[n] = aDataZ[n] - 1;//恢复
			}
			aOffset[n] = face;
		}
	}
	free(flagA);
	free(J);
}

__host__ void JacobiCPU(float *aDataX, float *aDataY, float *aDataZ,
	float *aTriX, float *aTriY, float *aTriZ,
	float *aObsX, float *aObsY, float *aObsZ,
	int iTriNum, int iDataNum, int iObsNum,
	float *result)
{
	int maxFaceNum = 6;
	float(*h_jbot)[NCOLUMN] = (float(*)[NCOLUMN])
		malloc(sizeof(float)* NCOLUMN * maxFaceNum*iDataNum);//申请内存空间//Gpu
	float(*h_jtop)[NCOLUMN] = (float(*)[NCOLUMN])
		malloc(sizeof(float)* NCOLUMN * maxFaceNum*iDataNum);//row=iDataNum*maxFaceNum //Gpu
	int *offset = (int*)malloc(sizeof(int)*iDataNum);//Gpu

	JacobiIndex(h_jtop, h_jbot, aTriX, aTriY, aTriZ, iTriNum,
		aDataX, aDataY, aDataZ, maxFaceNum, iDataNum, offset);

	int star = 0;
	for (int i = 0; i < iDataNum; i++)
	{
		star = i* maxFaceNum;
		for (int j = 0; j < iObsNum; j++)
		{
			result[j*iDataNum + i] =
				-G*P*(lineMethod(aObsX[j], aObsY[j], aObsZ[j], h_jtop, star, star + offset[i])
				- lineMethod(aObsX[j], aObsY[j], aObsZ[j], h_jbot, star, star + offset[i]));
			//printf("%f\t", Jaco[j*iDataNum + i]);	
		}
		//printf("\n");
	}

	free(h_jbot);
	free(h_jtop);
	free(offset);
}

__host__ void getSolve(int iDataNum, int iObsNum, float lambda,
	float*J, float* aObsG, float* aDataZ, float* aForwardG)
{
	float(*Jt) = (float*)malloc(sizeof(float)*iObsNum*iDataNum);
	float *Jtd = (float*)malloc(sizeof(float)*iDataNum);
	float *d = (float*)malloc(sizeof(float)*iObsNum);
	float *dp = (float*)malloc(sizeof(float)*iDataNum);
	float **H = (float**)malloc(sizeof(float*)*iDataNum);
	for (int i = 0; i < iDataNum; i++)
		H[i] = (float*)malloc(sizeof(float)*iDataNum);

	for (int i = 0; i < iObsNum; i++)
		d[i] = aObsG[i] - aForwardG[i];

	//for (int i = 0; i < iObsNum; i++)
	//{
	//	printf("%f\n",d[i]);
	//}
	//for (int i = 0; i < iDataNum; i++)
	//{
	//	for (int j = 0; j < iObsNum; j++)
	//	{
	//		printf("%f\t", J[j*iDataNum + i]);
	//	}
	//	printf("\n");
	//}

	for (int i = 0; i < iDataNum; i++)
	{
		for (int j = 0; j < iObsNum; j++)
		{
			Jt[j + i*iObsNum] = J[j*iDataNum + i];
		}
	}

	for (int i = 0; i < iDataNum; i++)
	{
		for (int j = 0; j < iDataNum; j++)
		{
			H[i][j] = 0.0;
		}
	}

	for (int i = 0; i < iDataNum; i++)
	{
		for (int j = 0; j < iDataNum; j++)
		{
			for (int k = 0; k < iObsNum; k++)
			{
				H[i][j] = H[i][j] + Jt[i*iObsNum + k] * J[k*iDataNum + j];
			}
		}
	}

	for (int j = 0; j < iDataNum; j++)
		H[j][j] += lambda;//inv
	//for (int i = 0; i < iDataNum; i++)
	//{
	//	for (int j = 0; j < iDataNum; j++)
	//	{
	//		printf("%f", H[i][j]);
	//	}
	//}

	for (int i = 0; i < iDataNum; i++)
		Jtd[i] = 0;

	for (int i = 0; i < iDataNum; i++)
	{
		for (int j = 0; j < iObsNum; j++)
		{
			Jtd[i] = Jtd[i] + Jt[i*iObsNum + j] * d[j];
		}
	}

	Matrix H_mat = { iDataNum, iDataNum, H };
	Matrix inv_H_mat = alloc_matrix(iDataNum, iDataNum);

	int a = destructive_invert_matrix(H_mat, inv_H_mat);
	if (a != 1)
	{
		printf("Matrix det =0");
	}

	for (int i = 0; i < iDataNum; i++)
	{
		dp[i] = 0;
	}

	for (int i = 0; i < iDataNum; i++)
	{
		for (int j = 0; j < inv_H_mat.cols; j++)
		{
			dp[i] = dp[i] + inv_H_mat.data[i][j] * Jtd[j];
		}
	}

	for (int i = 0; i < iDataNum; i++)
	{
		aDataZ[i] = aDataZ[i] + dp[i];
		//printf("%f\n", aDataZ[i]);
	}
}

__host__ void getGPUInfo()
{
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	int dev;
	for (dev = 0; dev < deviceCount; dev++)
	{
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);
		if (dev == 0)
		{
			if (/*deviceProp.major==9999 && */deviceProp.minor = 9999 && deviceProp.major == 9999)
				printf("\n");

		}
		printf("\nDevice%d:\"%s\"\n", dev, deviceProp.name);
		printf("Total amount of global memory                   %u bytes\n",
			deviceProp.totalGlobalMem);
		printf("Number of mltiprocessors                        %d\n",
			deviceProp.multiProcessorCount);
		printf("Total amount of constant memory:                %u bytes\n",
			deviceProp.totalConstMem);
		printf("Total amount of shared memory per block         %u bytes\n",
			deviceProp.sharedMemPerBlock);
		printf("Total number of registers available per block:  %d\n",
			deviceProp.regsPerBlock);
		printf("Warp size                                       %d\n",
			deviceProp.warpSize);
		printf("Maximum number of threada per block:            %d\n",
			deviceProp.maxThreadsPerBlock);
		printf("Maximum sizes of each dimension of a block:     %d x %d x %d\n",
			deviceProp.maxThreadsDim[0],
			deviceProp.maxThreadsDim[1],
			deviceProp.maxThreadsDim[2]);
		printf("Maximum size of each dimension of a grid:       %d x %d x %d\n",
			deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
		printf("Maximum memory pitch :                          %u bytes\n",
			deviceProp.memPitch);
		printf("Texture alignmemt                               %u bytes\n",
			deviceProp.texturePitchAlignment);
		printf("Clock rate                                      %.2f GHz\n",
			deviceProp.clockRate*1e-6f);
	}
	printf("\nTest PASSED\n");
}

__host__ void getTopG(const char *chFileName,
	int iObsNum, float aObsX[], float aObsY[], float aObsZ[],
	float *out)
{
	FILE *fp;
	fp = fopen(chFileName, "r");
	if (fp == NULL)
	{
		printf("fail to open the file！\n");
	}
	int iBlnNum = 0;
	float Temp;
	while (fscanf(fp, "%f", &Temp) != EOF)
	{
		iBlnNum++;
	}
	iBlnNum = iBlnNum / 2;
	//printf("%d\n",iBlnNum);
	fclose(fp);

	fp = fopen(chFileName, "r");

	float *aBlnX = (float*)malloc(sizeof(float)*iBlnNum);
	float *aBlnY = (float*)malloc(sizeof(float)*iBlnNum);

	for (int i = 0; i < iBlnNum; i++)
	{
		fscanf(fp, "%f   %f", &aBlnX[i], &aBlnY[i]);
	}
	fclose(fp);

	/*for (int i = 0; i < iBlnNum; i++)
	{
	printf("%f\t  %f\n",aBlnX[i],aBlnY[i]);
	}*/

	float dnx = 0;
	float dny = 0;
	float dnz = 1.0;


	for (int j = 0; j < iObsNum; j++)
	{

		//float a1 = index[j][3] - index[j][0],
		//	a2 = index[j][4] - index[j][1],
		//	a3 = index[j][5] - index[j][2];
		//float b1 = index[j][6] - index[j][3],
		//	b2 = index[j][7] - index[j][4],
		//	b3 = index[j][8] - index[j][5];

		//float dnx = CROSSX(a1, a2, a3, b1, b2, b3);
		//float dny = CROSSY(a1, a2, a3, b1, b2, b3);
		//float dnz = CROSSZ(a1, a2, a3, b1, b2, b3);

		float lineGravity = 0;
		for (int i = 0; i < iBlnNum - 1; i++)
		{

			float a1 = aBlnX[i],
				a2 = aBlnY[i],
				a3 = 0;
			float b1 = aBlnX[i + 1],
				b2 = aBlnY[i + 1],
				b3 = 0;

			float L = NORM(a1 - b1, a2 - b2, a3 - b3);

			float dtx = (b1 - a1) / L;
			float dty = (b2 - a2) / L;
			float dtz = (b3 - a3) / L;

			float dhx = CROSSX(dtx, dty, dtz, dnx, dny, dnz);
			float dhy = CROSSY(dtx, dty, dtz, dnx, dny, dnz);
			float dhz = CROSSZ(dtx, dty, dtz, dnx, dny, dnz);

			float vr1x = a1 - aObsX[j];
			float vr1y = a2 - aObsY[j];
			float vr1z = a3 - aObsZ[j];

			float vr2x = b1 - aObsX[j];
			float vr2y = b2 - aObsY[j];
			float vr2z = b3 - aObsZ[j];

			float v = DOT(dnx, dny, dnz, vr1x, vr1y, vr1z);
			float h = DOT(dhx, dhy, dhz, vr2x, vr2y, vr2z);
			float r1 = NORM(vr1x, vr1y, vr1z);
			float r2 = NORM(vr2x, vr2y, vr2z);

			float DN = L / (r2 + r1);

			float C = 2 * atanh(DN);
			float E = 0.5*(r1 + r2 - L*DN);
			float Q = (SIGN(v)) * 2 * atan(h*DN / (abs(v) + E));

			lineGravity = lineGravity + dnz*(h*C - v*Q);
		}
		out[j] = lineGravity;
	}



	free(aBlnX);
	free(aBlnY);
}

__host__ void BottCPU(float *aDataX, float *aDataY, float *aObsG,
	float *aTriX, float *aTriY, float *aTriZ,
	int iTriNum, int iDataNum,
	int iterate, float *aTopG,
	float *aForwardG, float *invZ, bool bFlag)
{
	int iObsNum = iDataNum;
	float *aObsX = (float*)malloc(sizeof(float)*iObsNum);
	float *aObsY = (float*)malloc(sizeof(float)*iObsNum);
	float *aObsZ = (float*)malloc(sizeof(float)*iObsNum);
	for (int i = 0; i < iDataNum; i++)
	{
		aObsX[i] = aDataX[i] + 0.001;
		aObsY[i] = aDataY[i] + 0.001;
		aObsZ[i] = 0;
	}
	float a = 0;
	for (int i = 0; i < iDataNum; i++)
	{
		invZ[i] = (aObsG[i] * P) / (41.98*P*P + a*aObsG[i]);
	}
	for (int i = 0; i < iterate; i++)
	{
		forwardGravityCPU(aDataX, aDataY, invZ,
			aTriX, aTriY, aTriZ,
			aObsX, aObsY, aObsZ,
			iTriNum, iObsNum, aTopG,
			aForwardG);

		for (int i = 0; i < iDataNum; i++)
		{
			invZ[i] = invZ[i] + (aObsG[i] - aForwardG[i]) / (2 * PI*P*G);
		}
		if (bFlag == true)
		{
			normPrint(aObsG, aForwardG, iObsNum);
		}
	}
	free(aObsX);
	free(aObsY);
	free(aObsZ);
}

__host__ void BottGPU(float *aDataX, float *aDataY, float *aObsG,
	float *aTriX, float *aTriY, float *aTriZ,
	int iTriNum, int iDataNum,
	int iterate, float *aTopG,
	float *aForwardG, float *invZ, bool bFlag)
{
	int iObsNum = iDataNum;
	float *aObsX = (float*)malloc(sizeof(float)*iObsNum);
	float *aObsY = (float*)malloc(sizeof(float)*iObsNum);
	float *aObsZ = (float*)malloc(sizeof(float)*iObsNum);
	for (int i = 0; i < iDataNum; i++)
	{
		aObsX[i] = aDataX[i] + 0.001;
		aObsY[i] = aDataY[i] + 0.001;
		aObsZ[i] = 0;
	}
	float a = 0;
	for (int i = 0; i < iDataNum; i++)
	{
		invZ[i] = (aObsG[i] * P) / (41.98*P*P + a*aObsG[i]);
	}
	for (int i = 0; i < iterate; i++)
	{
		forwardGravityGPU(aDataX, aDataY, invZ,
			aTriX, aTriY, aTriZ,
			aObsX, aObsY, aObsZ,
			iTriNum, iObsNum, aTopG,
			aForwardG);

		for (int i = 0; i < iDataNum; i++)
		{
			invZ[i] = invZ[i] + (aObsG[i] - aForwardG[i]) / (2 * PI*P*G);
		}
		if (bFlag == true)
		{
			normPrint(aObsG, aForwardG, iObsNum);
		}
	}
	free(aObsX);
	free(aObsY);
	free(aObsZ);
}

__host__ void NolineCPU(float *aDataX, float *aDataY, float *invZ,
	float *aTriX, float *aTriY, float *aTriZ,
	float *aObsX, float *aObsY, float *aObsZ, float *aObsG,
	int iTriNum, int iDataNum, int iObsNum,
	int iterate, float intiValue, float lambda, float *aTopG,
	float *aForwardG, bool bFlag)
{
	float *J = (float*)malloc(sizeof(float)*iObsNum*iDataNum);
	for (int i = 0; i < iDataNum; i++)
	{
		invZ[i] = intiValue;
	}
	for (int i = 0; i < iterate; i++)
	{
		forwardGravityCPU(aDataX, aDataY, invZ,
			aTriX, aTriY, aTriZ,
			aObsX, aObsY, aObsZ,
			iTriNum, iObsNum, aTopG,
			aForwardG);
		JacobiCPU(aDataX, aDataY, invZ,
			aTriX, aTriY, aTriZ,
			aObsX, aObsY, aObsZ,
			iTriNum, iDataNum, iObsNum,
			J);
		getSolve(iDataNum, iObsNum, lambda,
			J, aObsG, invZ, aForwardG);
		if (bFlag == true)
		{
			normPrint(aObsG, aForwardG, iObsNum);
		}
	}
	free(J);
}

__host__ void NolineGPU(float *aDataX, float *aDataY, float *invZ,
	float *aTriX, float *aTriY, float *aTriZ,
	float *aObsX, float *aObsY, float *aObsZ, float *aObsG,
	int iTriNum, int iDataNum, int iObsNum,
	int iterate, float intiValue, float lambda, float *aTopG,
	float *aForwardG,bool bFlag)
{
	float *J = (float*)malloc(sizeof(float)*iObsNum*iDataNum);
	for (int i = 0; i < iDataNum; i++)
	{
		invZ[i] = intiValue;
	}
	for (int i = 0; i < iterate; i++)
	{
		forwardGravityGPU(aDataX, aDataY, invZ,
			aTriX, aTriY, aTriZ,
			aObsX, aObsY, aObsZ,
			iTriNum, iObsNum, aTopG,
			aForwardG);
		JacobiCPU(aDataX, aDataY, invZ,
			aTriX, aTriY, aTriZ,
			aObsX, aObsY, aObsZ,
			iTriNum, iDataNum, iObsNum,
			J);
		getSolve(iDataNum, iObsNum, lambda,
			J, aObsG, invZ, aForwardG);
		if (bFlag == true)
		{
			normPrint(aObsG, aForwardG, iObsNum);
		}
	}
	free(J);
}

__global__ void Kernelreduce(const float *a, float *r)
{
	__shared__ float cache[BLOCK_SIZE];
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int cacheIndex = threadIdx.x;

	// copy data to shared memory from global memory
	cache[cacheIndex] = a[tid];
	__syncthreads();

	// add these data using reduce
	for (int i = blockDim.x / 2; i > 0; i /= 2)
	{
		if (cacheIndex < i)
		{
			cache[cacheIndex] += cache[cacheIndex + i];
		}
		__syncthreads();
	}

	// copy the result of reduce to global memory
	if (cacheIndex == 0)
		r[blockIdx.x] = cache[cacheIndex];
}

__global__ void KernelTop(float *midResult, float *d_obsX, float *d_obsY, float *d_obsZ,
	float *d_Arraytop,
	const int triNum, const int PointPerThreads)
{
	//__shared__ int sdata[128];
	int idx = (blockIdx.x*blockDim.x) + threadIdx.x;//iTriNum
	int obsIndex = idx / PointPerThreads;
	int triIndex = idx % PointPerThreads;
	//int cacheIndex = threadIdx.x;

	if (triIndex<triNum)
	{
		midResult[idx] = (double)lineMethod(d_obsX[obsIndex], d_obsY[obsIndex], d_obsZ[obsIndex], d_Arraytop, triIndex);
	}
}

void forwardGravityGPU(float *aDataX, float *aDataY, float *aDataZ,
	float *aTriX, float *aTriY, float *aTriZ,
	float *aObsX, float *aObsY, float *aObsZ,
	int iTriNum, int iObsNum, float topG[],
	float *Forward)
{
	int PointPerblock = (iTriNum%BLOCK_SIZE>0) ? (iTriNum / BLOCK_SIZE + 1) : (iTriNum / BLOCK_SIZE);
	int blockNum = iObsNum*PointPerblock;
	dim3 dimgrid(blockNum), dimblock(BLOCK_SIZE);


	int PointPerThreads = PointPerblock*BLOCK_SIZE;
	int midResultNum = blockNum*BLOCK_SIZE;
	int resultNum = iObsNum*PointPerblock;

	//printf("PointPerblock:  %d\n", PointPerblock);
	//printf("test: %d  %d\n", iTriNum, PointPerblock*BLOCK_SIZE);
	//printf("trinum*obsnum:  %d\n", iTriNum*iObsNum);
	//printf("blockNum*BLOCK_SIZE:  %d\n", blockNum*BLOCK_SIZE);
	//printf("blockNum: %d \n", blockNum);
	//printf("resultNum: %d \n", resultNum);

	const int result_bytes = sizeof(float)*resultNum;
	const int Array_bytes = sizeof(float)*iObsNum;
	const int midResult_bytes = sizeof(float)*midResultNum;
	const int FaceIndex_bytes = 12 * iTriNum * sizeof(float);

	float *result = (float*)malloc(result_bytes);
	//float *midResult = (float*)malloc(midResult_bytes);

	float(*h_indexTop)[NCOLUMN] = (float(*)[NCOLUMN])malloc(sizeof(float)* NCOLUMN * iTriNum);
	//float(*h_indexBot)[NCOLUMN] = (float(*)[NCOLUMN])malloc(sizeof(float)* NCOLUMN * iTriNum);

	forwardIndex(h_indexTop,aTriX, aTriY, aTriZ, iTriNum, aDataX, aDataY, aDataZ);

	float* d_obsX;
	float* d_obsY;
	float* d_obsZ;
	float* d_Arraytop;
	//float* d_Arraybot;

	float* d_midResult;
	float* d_result;

	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);

	cudaStatus = cudaMalloc((void**)&d_obsX, Array_bytes);
	cudaStatus = cudaMalloc((void**)&d_obsY, Array_bytes);
	cudaStatus = cudaMalloc((void**)&d_obsZ, Array_bytes);
	cudaStatus = cudaMalloc((void**)&d_Arraytop, FaceIndex_bytes);
	//cudaStatus = cudaMalloc((void**)&d_Arraybot, FaceIndex_bytes);

	cudaStatus = cudaMemcpy(d_obsX, aObsX, Array_bytes, cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(d_obsY, aObsY, Array_bytes, cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(d_obsZ, aObsZ, Array_bytes, cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(d_Arraytop, h_indexTop, FaceIndex_bytes, cudaMemcpyHostToDevice);
	//cudaStatus = cudaMemcpy(d_Arraybot, h_indexBot, FaceIndex_bytes, cudaMemcpyHostToDevice);

	cudaStatus = cudaMalloc((void**)&d_midResult, midResult_bytes);
	cudaStatus = cudaMalloc((void**)&d_result, result_bytes);

	KernelTop << < dimgrid, dimblock >> >
		(d_midResult, d_obsX, d_obsY, d_obsZ, d_Arraytop,
		iTriNum, PointPerThreads);
	Kernelreduce << < dimgrid, dimblock >> >
		(d_midResult, d_result);


	cudaStatus = cudaMemcpy(result, d_result, result_bytes, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	for (int i = 0; i < iObsNum; i++)
	{
		float sum = 0;
		for (int j = 0; j < PointPerblock; j++)
		{
			sum = sum + result[j + i*PointPerblock];
		}
		Forward[i] = -(sum + topG[i])*P*G;
	}

	//free(h_indexTop);
	cudaFree(d_Arraytop);
	cudaFree(d_obsX);
	cudaFree(d_obsY);
	cudaFree(d_obsZ);
	cudaFree(d_midResult);
	cudaFree(d_result);

}

__host__ void forwardGravityCPU(float *aDataX, float *aDataY, float *aDataZ,
	float *aTriX, float *aTriY, float *aTriZ,
	float *aObsX, float *aObsY, float *aObsZ,
	int iTriNum, int iObsNum, float aTopG[],
	float *out)
{
	const int iRows = iTriNum;
	float(*h_indexTop)[NCOLUMN] = (float(*)[NCOLUMN])malloc(sizeof(float)* NCOLUMN * iRows);
	//float(*h_indexBot)[NCOLUMN] = (float(*)[NCOLUMN])malloc(sizeof(float)* NCOLUMN * iRows);
	forwardIndex(h_indexTop, aTriX, aTriY, aTriZ, iTriNum, aDataX, aDataY, aDataZ);

	for (int i = 0; i < iObsNum; i++)
	{
		//out[i] = lineMethod(aObsX[i], aObsY[i], aObsZ[i], h_indexBot, 0, iTriNum);
		out[i] = -(lineMethod(aObsX[i], aObsY[i], aObsZ[i], h_indexTop, 0, iTriNum) + aTopG[i])*P*G;
	}

	//free(h_indexBot);
	free(h_indexTop);
}