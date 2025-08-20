#pragma once

class LogReg {
    private:
        int max_epochs_;
        int batch_size_;
        double learning_rate_;
        int patience_;
        double lambda_; // L2 regularization
        // Need an instance of DataManager
        // Need an instance of NeuralNetwork

    public:
        // Constructors
        LogReg();
        LogReg(int max_epochs, int batch_size, double lr, int patience, double lambda);

        // Perform Logistic Regression
        void train();
        void evaluate();
        void predict();
    
    private:
        double loss();

};