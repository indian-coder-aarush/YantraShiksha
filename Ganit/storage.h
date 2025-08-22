#pragma once


#include <iostream>
#include <vector>


// Utility functions
int offset(std::vector<int>& stride, std::vector<int>& index);
bool increment(std::vector<int>& index, std::vector<int>& shape);


// Main storage class
class storage {
private:
   void print_data(int depth,  std::vector<int> &index);


public:
   std::vector<int> shape;
   double* data;
   int size;
   std::vector<int> stride;


   storage();  // Default constructor
   storage(const std::vector<int> &dim, double default_value); // Param constructor
   storage(const storage& other);  // Copy constructor
   storage& operator=(const storage& other);  // Copy assignment
   ~storage();  // Destructor
   storage copy();


   void set_stride(const std::vector<int> &shape);


   std::vector<int> dimensions();
   double access(const std::vector<int> &indices);
   void change_value(const std::vector<int> &indices, double &value);
   void setslice(std::vector<std::vector<int>>& slice,storage& other);
   storage slice(std::vector<std::vector<int>>& slice);
   void print();
};


// Element-wise binary operations
storage operator+(storage& a, storage& b);
storage operator-(storage& a, storage& b);
storage operator*(storage& a, storage& b);
storage operator/(storage& a, storage& b);
storage operator^(storage &a, double power);
storage relu(storage &a);


// Transpose
storage T_s(storage& a);
storage T_s(storage& a, std::vector<int>& order);


// Other operations
storage sqrt(storage &a);
storage s_sin(storage &a, int terms);
storage s_cos(storage &a, int terms);
storage s_matmul(storage &a, storage &b);
storage convolution_s(storage &a,storage &b,int stride);
storage log_s(storage &a);
double dot(storage &a, storage &b);