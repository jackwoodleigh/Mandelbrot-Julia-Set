#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

extern void matToImage(char* filename, int* mat, int* dims);
extern void matToImageColor(char* filename, int* mat, int* dims);

int main(int argc, char **argv){

    int rank, numranks;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numranks);

    omp_set_num_threads(6);
    double start = MPI_Wtime();
    
    
    //double ranktime;
    //double* summedranktime = (double*) calloc(numranks, sizeof(double));
    
    

    int nx=2400; 
    int ny=1600;   
    double x0=0;
    double y0=0;
    double xStart=-2;
    double xEnd=1;
    double yStart=-1; //-1 and 1
    double yEnd=1;
    int chunk_size = ny*5;
    int julia_set = 0;


    // master worker 
    if(rank == 0){
        int* matrix=(int*)malloc(nx*ny*sizeof(int));
        
        
        int num_running = numranks-1;
        int init = 0;
        int current = 0;
        int total = nx*ny;


        // message structure run/end msg | start range | end range | respective values
        int* message = (int*) malloc(2*sizeof(int)); 

         
        int* recv_data = (int*) malloc((chunk_size + 2)*sizeof(int));
        MPI_Status status;
        int source_rank;


        while(num_running != 0){
            if (init < num_running){
                MPI_Recv(recv_data, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
                init++;
            }
            else{
                MPI_Recv(recv_data, (chunk_size + 2), MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
            }
            source_rank = status.MPI_SOURCE;
            

            // interact with data recieved 
            if(recv_data[0]){

                // for the range node computed set matrix values
                for(int i=recv_data[1]; i<recv_data[1]+chunk_size; i++){
                    matrix[i] = recv_data[(i-recv_data[1]) + 2];
                }
            }

            // send data if remaining 
            if(current < total){

                message[0] = 1; 
                message[1] = current;
                MPI_Send(message, 2, MPI_INT, source_rank, 0, MPI_COMM_WORLD);
                current += chunk_size;

            }
            // else send end msg 
            else{
                message[0] = 0;
                message[1] = 0;
                MPI_Send(message, 2, MPI_INT, source_rank, 0, MPI_COMM_WORLD);
                num_running--;

            }
        }
        

        double end = MPI_Wtime();
        printf("Time: %f", end-start);
        int dims[2];
        dims[0]=ny;
        dims[1]=nx;

    

        //matToImage("mandelbrot.jpg", matrix, dims);      
        matToImageColor("mandelbrotcolor.jpg", matrix, dims); 

    }


    // worker nodes 
    else{
        double* threadtime = (double*) calloc(omp_get_max_threads(), sizeof(double));
        double nodestart = MPI_Wtime();



        int maxIter=255;
        int row;
        int col;
        double x;
        double y;
        int iter;
        double temp;
        double threadstart;
        double threadend;

        int* recv_data = (int*) malloc(2*sizeof(int));
        int* message = (int*) malloc((chunk_size + 2)*sizeof(int));

        int startmsg = 0; 
        MPI_Send(&startmsg, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

        double cx = -0.4;
        double cy = 0.6;

        
        while(1){
            MPI_Recv(recv_data, 2, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            
            if(recv_data[0]){
                
                message[0] = 1;
                message[1] = recv_data[1];

                
                #pragma omp parallel for schedule(dynamic) private(row, col, x0, y0, x, y, iter, temp, threadstart, threadend)
                for(int i=recv_data[1]; i<recv_data[1]+chunk_size; i++){
                    threadstart = omp_get_wtime();
                    

                    row = i/nx;
                    col = i%nx;

                    x0=xStart+(1.0*col/nx)*(xEnd-xStart); // real part of z
                    y0=yStart+(1.0*row/ny)*(yEnd-yStart); // imaginary part of z

                    if(julia_set){
                        x=x0;
                        y=y0;
                        x0 = cx;
                        y0 = cy;
                    }
                    else{
                        x=0;
                        y=0;
                    }
                    
                    iter=0;

                    
                    while(iter<maxIter){
                        iter++;
                        temp = x*x - y*y + x0; // real part
                        y = 2*x*y + y0; // imaginary part
                        x = temp;
                        
                        
                        
                        if(x*x+y*y>4){
                            break;
                        }

                        
                    }
                    message[(i-recv_data[1]) + 2] = iter;
                    

                    
                    
                    threadend = omp_get_wtime();
                    #pragma omp critical
                    {
                        threadtime[omp_get_thread_num()] += threadend - threadstart;
                    }
                    
                    
                    
                }
                
                
                MPI_Send(message, (chunk_size + 2), MPI_INT, 0, 0, MPI_COMM_WORLD);
                

            }

            else{
                break;
            }
        }

        
        if (rank == 1){
            int n = omp_get_max_threads();
            double total = 0.0;
            for(int i=0; i<n; i++){
                printf("\n Thread: %d, time: %f\n", i, threadtime[i]);
            }
        }
        
        //double nodeend = MPI_Wtime();
        //ranktime = nodeend - nodestart;
    }
    
 //MPI_Reduce(threadtime, summedthreadtime, omp_get_max_threads(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    /*
    MPI_Gather(&ranktime, 1, MPI_DOUBLE, summedranktime, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 1){
        int n = numranks;
        double total = 0.0;
        for(int i=0; i<n; i++){
            printf("\n Thread: %d, time: %f\n", i, summedranktime[i]);
            
        }
       
    }

    */
    
    

    MPI_Finalize();
    return 0;
}
