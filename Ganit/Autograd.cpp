#include <iostream>
#include <vector>
#include <cmath>
#include"storage.h"
#include "Tanitra.h"
#include "Autograd.h"

Node::Node(){}

void Node::apply(storage &grad){
    accumulate_gradient(grad);
}

void Node::accumulate_gradient(storage& grad) {
    if (gradient.data == nullptr){
        gradient = grad;}
    else{
        gradient = grad + gradient;}
    }

void AddNode::apply(storage &grad)  {
        accumulate_gradient(grad);
        a->apply(grad);
        b->apply(grad);
    }

void SubNode::apply(storage &grad) {
        accumulate_gradient(grad);
        a->apply(grad);
        storage minus_ones(grad.shape,-1);
        storage grad_b = grad * minus_ones;
        b->apply(grad_b);
    }

void MulNode::apply(storage &grad) {
        accumulate_gradient(grad);
        storage grad_a = b->tensor->data * grad;
        storage grad_b = a->tensor->data * grad;
        a->apply(grad_a);
        b->apply(grad_b);
    }

void DivNode::apply(storage &grad) {
        accumulate_gradient(grad);
        storage minus_ones(grad.shape, -1);
        storage grad_a = grad / b->tensor->data;
        storage grad_b =  (grad * minus_ones * a->tensor->data)/(b->tensor->data^2);
        a->apply(grad_a);
        b->apply(grad_b);
    }

void SinNode::apply(storage &grad) {
        accumulate_gradient(grad);
        storage grad_a = s_cos(a->tensor->data,10) * grad;
        a->apply(grad_a);
    }

void CosNode::apply(storage &grad)  {
        accumulate_gradient(grad);
        storage minus_ones(grad.shape, -1);
        storage grad_a = s_sin(a->tensor->data,10) * grad * minus_ones;
        a->apply(grad_a);
    }

void ReshapeNode::apply(storage &grad){
        accumulate_gradient(grad);
        grad.shape = initial_shape;
        a->apply(grad);
    }

void MatmulNode::apply(storage &grad){
        accumulate_gradient(grad);
        storage grad_a = s_matmul(grad,T_s(b->tensor->data));
        storage grad_b = s_matmul(T_s(a->tensor->data),grad);
        a->apply(grad_a);
        b->apply(grad_b);
    }

void TNode::apply(storage &grad){
    accumulate_gradient(grad);
    storage grad_a = T_s(grad);
    a->apply(grad_a);
}

void SetItemNode::apply(storage &grad){
    accumulate_gradient(grad);
    storage grad_a = grad.slice(slice);
    grad_a = grad_a.copy();
    storage zeros(a->tensor->data.shape,0);
    grad.setslice(slice,zeros);
    a->apply(grad_a);
    b->apply(grad);
}

void GetItemNode::apply(storage &grad){
    accumulate_gradient(grad);
    storage zeros(a->tensor->data.shape,0);
    zeros.setslice(slice,grad);
    storage grad_a = zeros.copy();
    a->apply(grad_a);
}