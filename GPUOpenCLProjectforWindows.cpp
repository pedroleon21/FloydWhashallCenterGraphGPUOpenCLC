#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include<time.h>

#include <CL/cl.h>

#define DATA_SIZE 12000
#define INF 99999
using namespace std;

clock_t inicio, fim;


const char* ProgramSource = "__kernel void parallel_fw(__global uint * pathDistanceBuffer,\n"\
" __global uint * pathBuffer, const unsigned int numNodes, const unsigned int pass) \n"\
"{ \n"\
"   int xValue = get_global_id(0); \n"\
"   int yValue = get_global_id(1); \n"\
"   int k = pass; \n"\
"   int oldWeight = pathDistanceBuffer[yValue * numNodes + xValue]; \n"\
"   int tempWeight = (pathDistanceBuffer[yValue * numNodes + k] + pathDistanceBuffer[k * numNodes + xValue]); \n"\
"   if (tempWeight < oldWeight){ \n"\
"       pathDistanceBuffer[yValue * numNodes + xValue] = tempWeight; \n"\
"   } \n"\
"} \n"\
"\n";


const char* GraphCenter = "__kernel void graphCenter(__global float * pathBuffer, \n"\
"__global float * soma, __global float * grau,__global float *media, const unsigned int numNodes, \n"\
"const unsigned int pass) {\n"\
"   int INF = 99999;\n"\
"   int xValue = pass; \n"\
"   int yValue = get_global_id(1); \n"\
"   if (xValue != yValue && pathBuffer[xValue * numNodes + yValue] != INF) {       \n"\
"       soma[xValue] = soma[xValue] + pathBuffer[xValue * numNodes * yValue];\n"\
"       grau[xValue]++; \n"\
"   }\n"\
"   media[xValue] = soma[xValue] / grau[xValue]; \n"\
"} \n"\
"\n";


int _KernelGraphCenter(int* path_dis_mat){ //DATA_SIZE seq_dis_mat
    cl_context context;
    cl_context_properties properties[3];
    cl_kernel kernel;
    cl_command_queue command_queue;
    cl_program program;
    cl_int err;
    cl_uint num_of_platforms = 0;
    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_uint num_of_devices = 0;
    cl_mem soma_buffer, grau_buffer,dis_mat,media_buffer;
    size_t global[2];
    size_t local[2];
    float* paths   = (float*)malloc(sizeof(float) * DATA_SIZE * DATA_SIZE);
    float* media   = (float*)malloc(sizeof(float) * DATA_SIZE);
    float* soma    = (float*)malloc(sizeof(float) * DATA_SIZE);
    float* grau    = (float*)malloc(sizeof(float) * DATA_SIZE);
    int block_size = 4;
    int i, num_passes;
    for (i = 0; i < DATA_SIZE * DATA_SIZE; i++) {
        paths[i] = (float) path_dis_mat[i];
    }
    for ( i = 0; i < DATA_SIZE; i++)
    {
        grau[i] = soma[i] = media[i]=0;
    }

    if (clGetPlatformIDs(1, &platform_id, &num_of_platforms) != CL_SUCCESS){
        printf("Getplatformid\n");
        return 4;
    }


    if (clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &num_of_devices) != CL_SUCCESS){
        printf("Getdevice_id\n");
        return 5;
    }


    properties[0] = CL_CONTEXT_PLATFORM; properties[1] = (cl_context_properties)platform_id; properties[2] = 0;

    context = clCreateContext(properties, 1, &device_id, NULL, NULL, &err);

    command_queue = clCreateCommandQueue(context, device_id, 0, &err);

    program = clCreateProgramWithSource(context, 1, (const char**)&GraphCenter, NULL, &err);
    if (err != CL_SUCCESS)
    {
        return 6;
    }
    if (clBuildProgram(program, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS){
        printf("Error building program\n");
        return 7;
    }

    kernel = clCreateKernel(program, "graphCenter", &err);

    dis_mat     = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * DATA_SIZE * DATA_SIZE, NULL, NULL);
    soma_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * DATA_SIZE            , NULL, NULL);
    grau_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * DATA_SIZE            , NULL, NULL);
    media_buffer= clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * DATA_SIZE            , NULL, NULL);

    clEnqueueWriteBuffer(command_queue,
        dis_mat, CL_TRUE, 0,
        sizeof(float) * DATA_SIZE * DATA_SIZE, paths, 0, NULL, NULL);

    clEnqueueWriteBuffer(command_queue,
        soma_buffer, CL_TRUE, 0,
        sizeof(float) * DATA_SIZE, soma, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue,
        grau_buffer, CL_TRUE, 0,
        sizeof(float) * DATA_SIZE, grau, 0, NULL, NULL);

    clEnqueueWriteBuffer(command_queue,
        media_buffer, CL_TRUE, 0,
        sizeof(float) * DATA_SIZE, media, 0, NULL, NULL);

    int temp = DATA_SIZE;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &dis_mat);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &soma_buffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &grau_buffer);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &media_buffer);
    clSetKernelArg(kernel, 4, sizeof(int)   , &temp);
    clSetKernelArg(kernel, 5, sizeof(int)   , &temp);
    global[0] = DATA_SIZE; //tamanho do dados
    global[1] = DATA_SIZE; //tamanho do dados
    local[0] = block_size;
    local[1] = block_size;
    num_passes = DATA_SIZE;

    for (i = 0; i < num_passes; i++){
        clSetKernelArg(kernel, 5, sizeof(int), &i);
        clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global, local, 0, NULL, NULL);
        clFlush(command_queue);
    }

    clFinish(command_queue);

    clEnqueueReadBuffer(command_queue, soma_buffer , CL_TRUE, 0, sizeof(float)   * DATA_SIZE , soma,  0, NULL, NULL);
    clEnqueueReadBuffer(command_queue, grau_buffer , CL_TRUE, 0, sizeof(float)   * DATA_SIZE , grau,  0, NULL, NULL);
    clEnqueueReadBuffer(command_queue, media_buffer, CL_TRUE, 0, sizeof(float)   * DATA_SIZE , media, 0, NULL, NULL);

    int node_central_id=0;
    float node_central_media = media[0];
    int node_central_grau = grau[0];
    for (i = 0; i < DATA_SIZE; i++)
    {
        if (media[i] < node_central_media)
        {
            node_central_media = media[i];
            node_central_grau = grau[i];
            node_central_id = i;
        }
        else if (media[i] == node_central_media)
        {
            if (grau[i] > node_central_grau)
            {
                node_central_media = media[i];
                node_central_grau = grau[i];
                node_central_id = i;
            }
        }
    }

    printf("\paralelo Matrix{\n O Node %d eh o de maior centralidade.\n", node_central_id);
    printf("Numero de Arestas: %d\n", node_central_grau);
    printf("Media das Distancias: %.2f\n\n}\n", (float)node_central_media);

    clReleaseMemObject(dis_mat);
    clReleaseMemObject(soma_buffer);
    clReleaseMemObject(grau_buffer);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    free(soma);
    free(grau);
    free(paths);
    return 0;
}

void calcula_centralidade_buffer(int *dist) { //DATA_SIZE*DATA_SIZE
    int i, j, * soma = (int*)malloc(DATA_SIZE * sizeof(int)), * grau = (int*)malloc(DATA_SIZE * sizeof(int));
    float node_central_id, node_central_media, node_central_grau;
    float *media = (float*)malloc(DATA_SIZE*sizeof(float));

    for (i = 0; i < DATA_SIZE; i++)
    {
        soma[i] = 0;
        grau[i] = 0;

        for (j = 0; j < DATA_SIZE; j++)
        {
            if (i != j && dist[i * DATA_SIZE + j] != INF)
            {
                soma[i] = soma[i] + dist[i * DATA_SIZE + j];
                grau[i]++;
            }
        }
        media[i] = (float)soma[i] / (float)grau[i];
    }


    node_central_media = media[0];
    node_central_grau = grau[0];

    for (i = 0; i < DATA_SIZE; i++)
    {
        if (media[i] < node_central_media)
        {
            node_central_media = media[i];
            node_central_grau = grau[i];
            node_central_id = i;
        }
        else if (media[i] == node_central_media)
        {
            if (grau[i] > node_central_grau)
            {
                node_central_media = media[i];
                node_central_grau = grau[i];
                node_central_id = i;
            }
        }
    }
    printf("sequencial\n");
    printf("\nBuffer:{\n O Node %d eh o de maior centralidade.\n", (int)node_central_id);
    printf("Numero de Arestas: %d\n",(int)node_central_grau);
    printf("Media das Distancias: %.2f\n\n}\n", (float)node_central_media);
    free(grau);
    free(soma);
}

void calcula_centralidade(int **dist)
{
    int i, j, * soma = (int*)malloc(DATA_SIZE * sizeof(int)), * grau = (int*)malloc(DATA_SIZE * sizeof(int));
    float node_central_id, node_central_media, node_central_grau;
    float* media = (float*)malloc(DATA_SIZE * sizeof(float));

    for (i = 0; i < DATA_SIZE; i++)
    {
        soma[i] = 0;
        grau[i] = 0;

        for (j = 0; j < DATA_SIZE; j++)
        {
            if (i != j && dist[i][j] != INF)
            {
                soma[i] = soma[i] + dist[i][j];
                grau[i]++;
            }
        }
        media[i] = (float)soma[i] / (float)grau[i];
    }

    node_central_media = media[0];
    node_central_grau = grau[0];

    for (i = 0; i < DATA_SIZE; i++)
    {
        if (media[i] < node_central_media)
        {
            node_central_media = media[i];
            node_central_grau = grau[i];
            node_central_id = i;
        }
        else if (media[i] == node_central_media)
        {
            if (grau[i] > node_central_grau)
            {
                node_central_media = media[i];
                node_central_grau = grau[i];
                node_central_id = i;
            }
        }
    }
    printf("\nSeq Matrix{\n O Node %d eh o de maior centralidade.\n",(int)node_central_id);
    printf("Numero de Arestas: %d\n",(int)node_central_grau);
    printf("Media das Distancias: %.2f\n\n}\n", (float)node_central_media);
    free(grau);
    free(soma);
}

void printMatrix(int dist[][DATA_SIZE])
{
    int i, j, k = 0;
    for (i = 0; i < DATA_SIZE; i++)
    {
        for (j = 0; j < DATA_SIZE; j++)
        {
            k++;
            if (dist[i][j] == INF)
                printf("%7s", "INF");
            else
                cout << dist[i][j] << " ";
        }
        cout << endl;
    }
}


void floydWarshall(int **dist)
{
    int i, j, k;

    for (k = 0; k < DATA_SIZE; k++)
    {
        for (i = 0; i < DATA_SIZE; i++)
        {
            for (j = 0; j < DATA_SIZE; j++)
            {
                if (dist[i][k] + dist[k][j] < dist[i][j])
                    dist[i][j] = dist[i][k] + dist[k][j];
            }
        }
    }
    //fim = clock();
    //printMatrix(dist);
}


int* alloca_buffer(int s) { //DATA_SIZE*DATA_SIZE
    return (int*)malloc(s * s * sizeof(int));
}
int ** allocate_martix(int NumberOfvert)
{
    int **matrix = (int**)malloc(NumberOfvert*sizeof(int*));
    for (int i = 0; i < NumberOfvert; i++) {
        matrix[i] = (int*)malloc(NumberOfvert* sizeof(int));
    }
    return matrix;
}
void free_matrix(int** matrix, int source_size) {
    for (int i = 0; i < source_size; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

int main(void)
{
    cl_context context;
    cl_context_properties properties[3];
    cl_kernel kernel;
    cl_command_queue command_queue;
    cl_program program;
    cl_int err;
    cl_uint num_of_platforms = 0;
    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_uint num_of_devices = 0;
    cl_mem path_dis_buffer, path_buffer;
    size_t global[2];
    size_t local[2];
    int block_size = 4;
    int i, num_passes;
    bool equal = true;
    int *path_dis_mat = alloca_buffer(DATA_SIZE);
    int *path_mat     = alloca_buffer(DATA_SIZE);
    int **seq_dis_mat = /*int**/allocate_martix(DATA_SIZE);
    //preenche a matriz
    for (i = 0; i < DATA_SIZE * DATA_SIZE; i++) {
        int x = rand() % 500;
        path_dis_mat[i] = !x && i % DATA_SIZE ? INF : x; //gera numeros aleatórios entre 1 e 500 com INF
    }
    //copia para matriz buffer
    for (i = 0; i < DATA_SIZE; i++)
    {
        for (int j = 0; j < DATA_SIZE; j++)
        {
            seq_dis_mat[i][j] = path_dis_mat[i * DATA_SIZE + j];
        }
    }

    for (cl_int i = 0; i < DATA_SIZE; ++i)
    {
        for (cl_int j = 0; j < i; ++j)
        {
            path_mat[i * DATA_SIZE + j] = 0;
            path_mat[j * DATA_SIZE + i] = 0;
        }
        path_mat[i * DATA_SIZE + i] = 0;
    }
    if (clGetPlatformIDs(1, &platform_id, &num_of_platforms) != CL_SUCCESS)
    {
        printf("Getplatformid\n");
        return 1;
    }


    if (clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &num_of_devices) != CL_SUCCESS)
    {
        printf("Getdevice_id\n");
        return 2;
    }

    properties[0] = CL_CONTEXT_PLATFORM;
    properties[1] = (cl_context_properties)platform_id;
    properties[2] = 0;

    context = clCreateContext(properties, 1, &device_id, NULL, NULL, &err);

    command_queue = clCreateCommandQueue(context, device_id, 0, &err);

    program = clCreateProgramWithSource(context, 1, (const char**)&ProgramSource, NULL, &err);

    if (clBuildProgram(program, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS)
    {
        printf("Error building program\n");
        return 3;
    }

    kernel = clCreateKernel(program, "parallel_fw", &err);

    path_dis_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * DATA_SIZE * DATA_SIZE, NULL, NULL);
    path_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * DATA_SIZE * DATA_SIZE, NULL, NULL);

    clEnqueueWriteBuffer(command_queue,
        path_dis_buffer, CL_TRUE, 0,
        sizeof(int) * DATA_SIZE * DATA_SIZE, path_dis_mat, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue,
        path_buffer, CL_TRUE, 0,
        sizeof(int) * DATA_SIZE * DATA_SIZE, path_mat, 0, NULL, NULL);


    int temp = DATA_SIZE;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &path_dis_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &path_buffer);
    clSetKernelArg(kernel, 2, sizeof(int), &temp);
    clSetKernelArg(kernel, 3, sizeof(int), &temp);
    global[0] = DATA_SIZE; //tamanho do dados
    global[1] = DATA_SIZE; //tamanho do dados
    local[0] = block_size;
    local[1] = block_size;
    num_passes = DATA_SIZE;
    cout << "comeco do calculo paralelizado com "<< DATA_SIZE << " Vertices" << endl;
    inicio = clock();
    for (i = 0; i < num_passes; i++)
    {
        clSetKernelArg(kernel, 3, sizeof(int), &i);
        clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global, local, 0, NULL, NULL);
        clFlush(command_queue);
    }

    clFinish(command_queue);

    clEnqueueReadBuffer(command_queue, path_dis_buffer, CL_TRUE, 0, sizeof(int) * DATA_SIZE * DATA_SIZE, path_dis_mat, 0, NULL, NULL);
    clEnqueueReadBuffer(command_queue, path_buffer, CL_TRUE, 0, sizeof(int) * DATA_SIZE * DATA_SIZE, path_mat, 0, NULL, NULL);
        
    fim = clock();
    double Tempo = (double)((fim - inicio) / CLK_TCK);
    cout << "Tempo para execucao paralelizado :" << Tempo << " Segundos" << endl;
    calcula_centralidade_buffer(path_dis_mat);


    //execução sequencial
    inicio = clock();
    floydWarshall(seq_dis_mat);
    fim = clock();
    Tempo = (double)((fim - inicio) / CLK_TCK);
    cout << "Tempo de execucao sequencial: " << Tempo << " Segundos" << endl;
    inicio = clock();
    calcula_centralidade(seq_dis_mat);
    fim = clock();
    Tempo = (double)((fim - inicio) / CLK_TCK);
    cout << "Tempo para o calculo do centro: " << Tempo <<" Segundos" <<endl;
    for (i = 0; i < DATA_SIZE; i++)
    {
        for (int j = 0; j < DATA_SIZE; j++)
        {
            if (seq_dis_mat[i][j] != path_dis_mat[i * DATA_SIZE + j]){
                equal = false;
            }
        }
    }
    cout << "Matrizes identicas: " << (equal ? "True" : "False") << endl;

    clReleaseMemObject(path_dis_buffer);
    clReleaseMemObject(path_buffer);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);
    free_matrix(seq_dis_mat, DATA_SIZE);
    free(path_dis_mat);
    free(path_mat);
    system("pause");
    return 0;
}
