#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
//using namespace std;
struct Car {
    double Age = 0.0;
    double Selling_Price = 0.0;
    double Present_Price = 0.0;
    double Kms_Driven = 0.0;
    double Fuel_Type = 0.0;
    double Seller_Type = 0.0;
    double Transmission = 0.0;
    double Owner = 0.0;
};
// 函数用于计算向量与矩阵的乘积
double Linear_Regression(const std::vector<double>& vector, const std::vector<std::vector<double>>& matrix) {
    int vectorSize = vector.size();
    int matrixCols = matrix[0].size();

    if (vectorSize != matrix.size() || matrixCols != 1) {
        std::cerr << "Invalid input dimensions for vector-matrix multiplication." << std::endl;
        return 0.0;
    }

    double result = 0.0;

    for (int i = 0; i < vectorSize; i++) {
        result += vector[i] * matrix[i][0];
    }

    return result;
}

Car getCarcsValue(int num) {
    std::ifstream inputFile;
    inputFile.open("C:/Users/user/Desktop/ML_HW2/py/Predict.csv", std::ios_base::in);
    if (!inputFile.is_open()) {
        std::cout << "Error , File Not Found" << std::endl;
        std::exit(EXIT_FAILURE);
        //return ;
    }
    std::string line, buffer;
    std::vector<Car> cars;

    while (std::getline(inputFile, line)) {
        std::istringstream iss(line);
        Car car;
        iss >> car.Age;
        iss.ignore(); // 忽略逗號
        iss >> car.Selling_Price;
        iss.ignore();
        iss >> car.Present_Price;
        iss.ignore();
        iss >> car.Kms_Driven;
        iss.ignore();
        iss >> car.Fuel_Type;
        iss.ignore();
        iss >> car.Seller_Type;
        iss.ignore();
        iss >> car.Transmission;
        iss.ignore();
        iss >> car.Owner;
        cars.push_back(car);
    }
    cars.erase(cars.begin());
    Car B= cars[num];
    return B;
}
int main() {
    std::cout<<"please enter number 0~286"<<std::endl;
    int num;
    std::cin >> num;  
    Car carsvalue= getCarcsValue(num);
    std::cout << carsvalue.Age;
    const std::vector<double> weight = { -0.0500, 0.6337, -0.1318, -0.2604, 0.1457 };
    std::vector<std::vector<double>> intputvalue = { {carsvalue.Age}, {carsvalue.Present_Price}, 
                                                    {carsvalue.Kms_Driven}, {carsvalue.Seller_Type}, {carsvalue.Fuel_Type} 
                                                    };
    double result = Linear_Regression(weight, intputvalue);
    double bias = 0.6716;
    result+= bias;
    std::cout << "intputvalue " << carsvalue.Age <<"  " << carsvalue.Present_Price << "  " << carsvalue.Kms_Driven << "  " << carsvalue.Seller_Type << "  " << carsvalue.Fuel_Type << std::endl;
    std::cout << "actual value: " << carsvalue.Selling_Price<<std::endl;
    std::cout << "predicted_value: " << result << std::endl;

    return 0;
}
