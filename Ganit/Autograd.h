#pragma once
#include <memory>
#include <vector>
#include "storage.h"
#include "Tanitra.h"

class Tensor;

class Node {
public:
    Node();
    std::shared_ptr<Tensor> tensor;
    storage gradient;

    void accumulate_gradient(storage& grad);
    virtual void apply(storage& grad);
};

class AddNode : public Node {
public:
    std::shared_ptr<Node> a, b;
    void apply(storage& grad) override;
};

class SubNode : public Node {
public:
    std::shared_ptr<Node> a, b;
    void apply(storage& grad) override;
};

class MulNode : public Node {
public:
    std::shared_ptr<Node> a, b;
    void apply(storage& grad) override;
};

class DivNode : public Node {
public:
    std::shared_ptr<Node> a, b;
    void apply(storage& grad) override;
};

class SinNode : public Node {
public:
    std::shared_ptr<Node> a;
    void apply(storage& grad) override;
};

class CosNode : public Node {
public:
    std::shared_ptr<Node> a;
    void apply(storage& grad) override;
};

class ReshapeNode : public Node {
public:
    std::shared_ptr<Node> a;
    std::vector<size_t> initial_shape;
    void apply(storage& grad) override;
};

class MatmulNode : public Node {
public:
    std::shared_ptr<Node> a, b;
    void apply(storage& grad) override;
};
