#include <iostream>
#include <vector>
#include <cmath>
#include"storage.h"


int offset(std::vector<int>& stride,std::vector<int>& index){
   int offset_index = 0;
   for(int i = 0;i<stride.size();i++){
       offset_index += stride[i]*index[i];
   }
   return offset_index;
}


bool increment(std::vector<int>& index,std::vector<int>& shape){
   for(int i = index.size()-1;i>=0;--i){
       if(index[i]+1<shape[i]){
           index[i]++;
           return true;
       }
       index[i] = 0;
   }
   return false;
}


void storage::print_data(int depth , std::vector<int> &index) {
   std::cout << '[' << ' ';
   for (int i = 0; i < shape[depth]; ++i) {
       index[depth] = i;
       if (depth == shape.size() - 1) {
           std::cout << this->access(index) << ' ';
       } else {
           print_data(depth + 1, index);
       }
   }
   std::cout << ']';
}


storage::storage(const storage& other)
     : shape(other.shape), size(other.size), data(other.data) ,
     stride(other.stride){
   }


void storage::set_stride(const std::vector<int> &shape){
   stride.resize(shape.size());
   int prod = 1;
   for (int i = shape.size() - 1; i >= 0; --i) {
       stride[i] = prod;
       prod *= shape[i];
   }
}


storage::storage():data(nullptr), size(0){}


storage& storage::operator=(const storage& other) {
       if (this != &other) {
           shape = other.shape;
           size  = other.size;
           data  = other.data;
           stride = other.stride;
       }
       return *this;
   }


storage::storage(const std::vector<int> &dim,double default_value) : shape(dim), data(nullptr), size(1) {
       for (const int &i:dim) {
           size *= i;
       }
       double* a = new double[size];
       data = a;
       for (int i = 0; i < size; i++) {
           *(data+i) = default_value;
       }
       set_stride(shape);
   }


storage::~storage() {
   }


storage storage::copy(){
   storage copy;
   copy.data = new double[size];
   copy.shape = shape;
   copy.stride = stride;
   copy.size = size;
   for(int i = 0; i < copy.size; i++){
       copy.data[i] = data[i];
   }
   return copy;
}


std::vector<int> storage::dimensions() {
       return shape;
   }


double storage::access(const std::vector<int> &indices) {
       int index = 0;
       for (int i = 0; i < indices.size(); i++) {
           index += stride[i]*indices[i];
       }
       return *(data+index);
   }


   void storage::change_value(const std::vector<int> &indices, double &value) {
       int index = 0;
       for (int i = 0; i < indices.size(); i++) {
           index += stride[i]*indices[i];
       }
       *(data+index) = value;
   }




   void storage::print() {
       std::vector<int> a(shape.size(),0);
       print_data(0,a);
   }




// Element-wise addition
template<typename Op>
storage elementwise_op(storage &a,storage &b,Op op) {
   storage result(a.dimensions(), 0);
   std::vector<int> index(a.shape.size(),0);
   int a_offset,b_offset,result_offset;
   do{
       a_offset = offset(a.stride,index);
       b_offset = offset(b.stride,index);
       result_offset = offset(result.stride,index);
       *(result.data+result_offset) = op(*(a.data+a_offset),*(b.data+b_offset));
   }while(increment(index,a.shape));
   return result;
}


storage operator+(storage& a, storage& b) {
   return elementwise_op(a, b, [](double x, double y) { return x + y; });
}


// Element-wise subtraction
storage operator-(storage& a, storage& b) {
   return elementwise_op(a, b, [](double x, double y) { return x - y; });
}




// Element-wise multiplication
storage operator*(storage& a, storage& b) {
   return elementwise_op(a, b, [](double x, double y) { return x * y; });
}




// Element-wise division
storage operator/(storage& a, storage& b) {
   return elementwise_op(a, b, [](double x, double y) { return x / y; });
}


storage T_s(storage& a){
   storage b(a);
   b.shape[0] = a.shape[1];
   b.shape[1] = a.shape[0];
   b.stride[0] = a.stride[1];
   b.stride[1] = a.stride[0];
   return b;
}


storage T_s(storage& a, std::vector<int>& order){
   std::vector<int> new_shape(a.shape.size(),0);
   std::vector<int> new_strides(a.stride.size(),0);
   for(int i = 0; i<a.shape.size();i++){
       new_strides[i] = a.stride[order[i]];
       new_shape[i] = a.shape[order[i]];
   }
   storage b(a);
   b.shape = new_shape;
   b.stride = new_strides;
   return b;
}


// Raise each element to a power
storage operator ^(storage &a, double power) {
   storage result(a.dimensions(), 0);
   for (int i = 0; i < a.size; i++) {
       result.data[i] = pow(a.data[i],power);
   }
   return result;
}

// Element-wise square root
storage sqrt(storage &a) {
   storage result(a.dimensions(), 0);
   for (int i = 0; i < a.size; i++) {
       *(result.data+i) = pow(*(a.data+i),0.5);
   }
   return result;
}


// Matrix multiplication
storage s_matmul(storage &a , storage &b){
   storage result({a.shape[0],b.shape[1]}, 0);
   for (int i = 0; i < a.shape[0]; i++) {
       for (int j = 0; j < b.shape[1]; j++) {
       double result_i_j = 0;
           for (int k = 0; k < b.shape[0]; k++) {
               result_i_j += a.data[i*a.stride[0]+k] * b.data[k*b.stride[0]+j];
           }
           result.data[i*result.stride[0]+j] =  result_i_j;
       }
   }
   return result;
}


// Dot product (1D tensors)
double dot(storage &a , storage &b) {
   double result = 0;
   for (int i = 0; i < a.shape[0]; i++) {
       result += a.data[i] * b.data[i];
   }
   return result;
}


// Element-wise sine using Taylor series
storage s_sin(storage &a,int terms) {
   storage return_variable = a;
   double result = 0;
   double fact = 1;
   int a_offset;
   std::vector<int> index(a.shape.size(),0);
   do {
       a_offset = offset(a.stride,index);
       for (int j = 0; j < terms ; j++) {
           fact = 1;
           for (int k = 1; k <= (2*j+1); k++) {
               fact *= k;
           }
           result += (pow((0-1),j)*pow((*(a.data+a_offset)),(2*j+1)))/fact;
       }
       *(return_variable.data+a_offset) = result;
       result = 0;
   }while(increment(index,a.shape));
   return return_variable;
}


// Element-wise cosine using Taylor series
storage s_cos(storage &a,int terms) {
   storage return_variable = a;
   double result = 0;
   float fact = 1;
   int a_offset;
   std::vector<int> index(a.shape.size(),0);
   do {
       a_offset = offset(a.stride,index);
       for (int j = 0; j < terms ; j++) {
           fact = 1;
           for (int k = 1; k <= (2*j); k++) {
               fact *= k;
           }
           result += (pow((0-1),j)*pow((*(a.data+a_offset)),(2*j)))/fact;
       }
       *(return_variable.data+a_offset) = result;
       result = 0;
   }while(increment(index,a.shape));
   return return_variable;
}


storage storage::slice(std::vector<std::vector<int>>& slice){
   std::vector<int> new_shape;
   std::vector<int> new_stride;
   int step;
   int index_offset = 0;
   for(int i=0;i<slice.size();i++){
       if(slice[i].size() != 1){
       if(slice[i][2] < 0){
          index_offset += stride[i]*(slice[i][1] > 0 ? slice[i][1]:shape[i]+slice[i][1]);
       }
       else{
          index_offset += stride[i]*(slice[i][0] > 0 ? slice[i][0]:shape[i]+slice[i][0]);
       }
       int len = 0;
       if ((slice[i][2] > 0 && slice[i][0] < slice[i][1]) || (slice[i][2] < 0 && slice[i][0] > slice[i][1])) {
           len = ((slice[i][1] - slice[i][0] + slice[i][2] + (slice[i][2] > 0 ? -1 : 1)) / slice[i][2]);
       }
       new_shape.push_back(len);
       new_stride.push_back(stride[i]*slice[i][2]);
       }
   }
   if(new_shape.size()!=shape.size()){
       for (int i = slice.size(); i < shape.size(); i++) {
       new_shape.push_back(shape[i]);
       new_stride.push_back(stride[i]);
   }
   }
   storage b;
   b.stride = new_stride;
   b.shape = new_shape;
   b.data = &data[index_offset];
   b.size = 1;
   for(int j = 0; j<b.shape.size();j++){
       b.size *= b.shape[j];
   }
   return b;
}


void storage::setslice(std::vector<std::vector<int>>& slice,storage& other){
   int index_offset = 0;
   std::vector<int> new_stride;
   for(int i=0;i<slice.size();i++){
       if(slice[i].size() != 1){
       new_stride.push_back(stride[i]*slice[i][2]);
       }
       index_offset += stride[i]*slice[i][0];
   }
   if(new_stride.size()!=stride.size()){
       for (int i = slice.size(); i < shape.size(); i++) {
       new_stride.push_back(stride[i]);
   }
   }
   int offset_assign;
   std::vector<int> index(other.shape.size(), 0);
   double* data_to_be_assigned = &data[index_offset];
   do{
       offset_assign = offset(new_stride, index);
       data[offset_assign] = other.access(index);
   }while(increment(index,other.shape));
}


storage convolution_s(storage a,storage b,int stride){
   std::vector<int> output_shape = {static_cast<int>((a.shape[0]-b.shape[0])/stride)+1,
   static_cast<int>((a.shape[1]-b.shape[1])/stride)+1};
   storage output(output_shape,0);
   double value_i_j = 0;
   int start_i = 0;
   int start_j = 0;
   std::vector<std::vector<int>> slice;
   storage window;
   for(int i = 0;i < output.shape[0];i++){
       start_j = 0;
       for(int j = 0; j < output.shape[1]; j++){
           value_i_j = 0;
           slice = {{start_i,start_i+b.shape[0],1},{start_j,start_j+b.shape[1],1}};
           window = a.slice(slice);
           for(int k = 0;k<window.shape[0];k++){
               for(int l = 0;l<window.shape[1];l++){
                   value_i_j+=window.data[k*window.stride[0]+l]*b.data[k*b.stride[0]+l];
               }
           }
           output.data[i*output.stride[0]+j] = value_i_j;
           start_j += stride;
       }
       start_i+=stride;
   }
   return output;
}

