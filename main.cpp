#include <iostream>
#include <string.h>
#include <cstdlib>
#include <mpi/mpi.h>

using namespace std;

#define BORDER "*********************************"

#define STD_M_SIZE 1000
#define STD_STEP 0.001
#define MAX_TIME 1.0
#define INIT_VAL 1.0

#define ROOT 0

void printBorder(){
    cout << BORDER << endl;
}

double * initVector(const int size){
    double * vector = new double [size];

    for (int i = 0; i < size; ++i){
        vector[i] = INIT_VAL;
    }

    return vector;
}

void printVector(const double * vector, const int size){
    for (int i = 0; i < size; ++i){
        cout << vector[i] << ' ';
    }
    cout << endl;
}

class Matrix{
private:
    double ** data;
    const int size;

public:
    Matrix(const int s): size(s){
        data = new double * [size];
        for (int i = 0; i < size; ++i){
            data[i] = new double[size];
        }
    }

    ~Matrix(){
        for (int i = 0; i < size; ++i){
            delete[] data[i];
        }
        delete[] data;
    }

    Matrix(const Matrix& other):size(other.size){
        data = new double* [size];
        for (int i = 0; i < size; ++i){
            data[i] = new double [size];
            memcpy(data[i], other.data[i], sizeof(double) * size);
        }
    }

    Matrix& fill(){
        data[0][0] = 2;
        data[0][1] = 1;
        data[1][0] = 3;
        data[1][1] = 4;
        return *this;
    }

    Matrix& fill1(){
        for (int i = 0; i < size; ++i){
            for (int j = 0; j < size; ++j){
                data[i][j] = 1;
            }
        }
        return *this;
    }

    Matrix& fillE(){
        srand((unsigned)time(NULL));
        for (int i = 0; i < size; ++i){
            data[i][i] = (rand() / (double)RAND_MAX - 0.5 * 100);
        }
        return *this;
    }

    Matrix& fillSimplest(){
        data[0][0] = 1;
        data[2][2] = 3;
        return *this;
    }

    Matrix& multiply(const double value){
        for (int i = 0; i < size; ++i){
            for (int j = 0; j < size; ++j){
                data[i][j] *= value;
            }
        }
        return *this;
    }

    Matrix& multiply(const Matrix& other){
        double ** tmp = new double* [size];
        for (int i = 0; i < size; ++i){
            tmp[i] = new double [size];
        }

        for (int i = 0; i < size; ++i){
            for (int j = 0; j < size; ++j){
                tmp[i][j] = 0;
                for (int k = 0; k < size; ++k){
                    tmp[i][j] += data[i][k] * other.data[k][j];
                }
            }
        }

        for (int i = 0; i < size; ++i){
            delete[] data[i];
        }
        delete[] data;

        data = tmp;

        return *this;
    }

    double * multiply(double * vector) const{
        double * res = new double [size];

        for (int i = 0; i < size; ++i){
            res[i] = 0;
            for (int k = 0; k < size; ++k){
                res[i] += data[i][k] * vector[k];
            }
        }

        return res;
    }

    double * partiallyMultiply(double * vector, const int startInd, const int endInt) const{
        double * res = new double [size];

        for (int i = startInd; i < endInt; ++i){
            res[i] = 0;
            for (int k = 0; k < size; ++k){
                res[i] += data[i][k] * vector[k];
            }
        }

        return res;
    }

    Matrix& addE(){
        for (int i = 0; i < size; ++i){
            data[i][i] += 1;
        }
        return *this;
    }

    const Matrix& print() const {
        printBorder();
        for (int i = 0; i < size; ++i){
            for (int j = 0; j < size; ++j){
                cout << data[i][j] << ' ';
            }
            cout << endl;
        }
        printBorder();
        cout.flush();
        return *this;
    }

    //A' = E + hA * ( E + ( h / 2) * A * ( E  + ( h / 3) * A * ( E + ( h / 4) * A )))
    Matrix& RungeOperator(const double h){
        const Matrix initial(*this);
        multiply(h / 4).addE();
        multiply(initial).multiply(h / 3).addE();
        multiply(initial).multiply(h / 2).addE();
        multiply(initial).multiply(h).addE();
        return *this;
    }
};

int main(int argc, char **argv) {
    const int size = STD_M_SIZE;
    double step = STD_STEP;

    int processNum, processCount;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);
    MPI_Comm_rank(MPI_COMM_WORLD, &processNum);

    double timeGenStart = MPI_Wtime();
    Matrix A(size);
    A.fillE();
    A.RungeOperator(step);
    double timeGenEnd = MPI_Wtime();

    if (processNum == ROOT){
        //cout << "Runge operator matrix: " << endl;
        cout << "time used to generate matrix: " << timeGenEnd - timeGenStart << " seconds." << endl;
        //A.print();
        cout.flush();
    }

    int rowCount = (size / processCount) + (size % processCount > processNum ? 1 : 0);
    int endInd;
    MPI_Scan(&rowCount, &endInd, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    int startInd = endInd - rowCount;

    int * displs = new int[processCount];
    int * sendcnts = new int[processCount];
    MPI_Allgather(&rowCount, 1, MPI_INT, sendcnts, 1, MPI_INT, MPI_COMM_WORLD);
    displs[0] = 0;
    for (int i = 1; i < processCount; ++i){
        displs[i] = displs[i-1] + sendcnts[i-1];
    }

//    cout << "Process: " << processNum << ". Start: " << startInd << ". End: " << endInd << endl;
//    cout.flush();
//
//    MPI_Barrier(MPI_COMM_WORLD);

    double * tmpVec = initVector(size);
    double time = 0;
    double transferTime = 0;
    double workStart = MPI_Wtime();
    while (time <= MAX_TIME){
//        if (processNum == ROOT){
//            cout << "Time: " << time << ". Values: ";
//            printVector(tmpVec, size);
//            cout.flush();
//        }

        double * buf = A.partiallyMultiply(tmpVec, startInd, endInd);
        delete[] tmpVec;
        tmpVec = buf;

        double transferStart = MPI_Wtime();
        MPI_Allgatherv(tmpVec + startInd, rowCount, MPI_DOUBLE, tmpVec, sendcnts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
        double transferEnd = MPI_Wtime();
        transferTime += transferEnd - transferStart;

        time += step;
    }
    double workEnd = MPI_Wtime();

    delete[] tmpVec;

    if (processNum == ROOT){
        cout << "Time used: " << workEnd - workStart << " seconds." << endl;
        cout << "Data transfer time: " << transferTime << "seconds." << endl;
        cout.flush();
    }

    MPI_Finalize();
    return 0;
}