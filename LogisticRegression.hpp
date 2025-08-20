#pragma once

#include <vector>


template<class T = double>
class LogReg {
    private:
        int max_epochs_;
        // Need an instance of DataManager
        // Need an instance of NeuralNetwork

    public:
        // Constructors
        LogReg(): max_epochs_(100) {} // Default number of epochs is 100
        LogReg(int max_epochs) : max_epochs_(max_epochs) {} // Set the maximum number of epochs for the regression.

        // Perform Logistic Regression
        void train() {
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

        void evaluate() {
            // Build a confusion matrix
            // Calculate Accuracy
            // Calculate Recall
            // Calculate Precision
            // Calculate F1
            // Calculate MCC
        }

        void predict() {
            // Calculate Z using NeuralNetwork
            // Calculate Y^hat using softmax in NeuralNetwork
        }

};