#include "LogisticRegression.hpp"

LogReg::LogReg()
    : max_epochs_(100), batch_size_(32), learning_rate_(0.01), patience_(5), lambda_(0.0) {};

LogReg::LogReg(int max_epochs, int batch_size, double lr, int patience, double lambda)
    : max_epochs_(max_epochs), batch_size_(batch_size), learning_rate_(lr), patience_(patience), lambda_(lambda) {}

    void LogReg::train() {
        // Shuffle the training data
                // Split the training data into mini batch sizes
                // For epoch = 1 to epoch = max_epochs
                    // For each batch
                        // forwardPass
                        // backwardPass
                    // After all batches complete, compute training loss and dev loss.
                    // Run Evaluation on dev loss
                    // Early stopping check
    }

    void LogReg::evaluate() {
        // Build a confusion matrix
        // Calculate Accuracy
        // Calculate Recall
        // Calculate Precision
        // Calculate F1
        // Calculate MCC
    }

    void LogReg::predict() {
        // Calculate Z using NeuralNetwork
        // Calculate Y^hat using softmax in NeuralNetwork
    }

    double LogReg::loss() {
        // Calculate the loss after an epoch has been completed
        return 0.0;
    }