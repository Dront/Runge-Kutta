#include <iostream>
#include <string.h>
#include <cstdlib>
#include <cmath>
//#include <mpi/mpi.h>

using namespace std;

#define BORDER "*********************************"

#define STD_M_SIZE 2
#define STD_STEP 0.0001
#define MAX_TIME 1.0
#define INIT_VAL 1.0

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
        for (int i = 0; i < size; ++i){
            data[i][i] = 1;
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
    int size = STD_M_SIZE;
    double step = STD_STEP;

    Matrix A(size);
    A.fill();
    A.RungeOperator(step);
    A.print();

    double * tmpVec = initVector(size);
    double time = 0;
    while (time <= MAX_TIME){
        cout << "Time: " << time << ". Values: ";
        printVector(tmpVec, size);
        double * buf = A.multiply(tmpVec);
        delete[] tmpVec;
        tmpVec = buf;
        time += step;
    }

    return 0;
}