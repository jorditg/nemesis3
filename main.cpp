#include <arrayfire.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <af/util.h>
#include <math.h>
#include "mnist.h"

using namespace af;
using std::vector;

float accuracy(const array& predicted, const array& target)
{
    array val, plabels, tlabels;
    max(val, tlabels, target, 1);
    max(val, plabels, predicted, 1);
    return 100 * count<float>(plabels == tlabels) / tlabels.elements();
}

// Activation function
array activation(const array &val)
{
    return 1.0f / (1.0f + exp(-val));
}

// Derivative of the activation function
array activation_deriv(const array &out)
{
    return out * (1.0f - out);
}

// Cost function
double error(const array &out,
             const array &pred)
{
    array dif = (out - pred);
    return 0.5*sum<float>(dif * dif);
}

class ann {

private:
    int num_layers;
    vector<array> weights;
    vector<array> weights_inc;

    // Add bias input to the output from previous layer
    array add_bias(const array &in);

    vector<array> forward_propagate(const array& input);

    void back_propagate(const vector<array> signal,
                        const array &pred,
                        const float &lr,
                        const float &momentum);
public:

    // Create a network with given parameters
    ann(vector<int> layers, double range=0.05);

    // Output after single pass of forward propagation
    array predict(const array &input);

    // Method to train the neural net
    float train(const array &input, const array &target,
                 const array &input_test, const array &target_test,
                 float lr = 0.1,
                 float momentum = 0.9,
                 int max_epochs = 100000,
                 int batch_size = 128,
                 float maxerr = 1.0,
                 int testing_interval = 100,
                 bool verbose = true);
};

inline array ann::add_bias(const array &in)
{
    // Bias input is added on top of given input
    return join(1, constant(1.0f, in.dims(0), 1), in);
}

vector<array> ann::forward_propagate(const array& input)
{
    // Get activations at each layer
    vector<array> signal(num_layers);
    signal[0] = input;

    for (int i = 0; i < num_layers - 1; i++) {
        array in = add_bias(signal[i]);
        array out = matmul(in, weights[i]);
        signal[i + 1] = activation(out);
    }

    return signal;
}

void ann::back_propagate(const vector<array> signal,
                         const array &target,
                         const float &lr,
                         const float &momentum)
{

    // Get error for output layer
    array out = signal[num_layers  - 1];
    array err = (out - target);
    int m = target.dims(0);
    const float alpha = lr / m;

    for (int i = num_layers - 2; i >= 0; i--) {
        array in = add_bias(signal[i]);
        array delta = (activation_deriv(out) * err).T();

        // Adjust weights
        array grad = -(alpha * matmul(delta, in));
        weights_inc[i] = momentum * weights_inc[i] + (1.0 - momentum) * grad.T();
        weights[i] += weights_inc[i];

        // Input to current layer is output of previous
        out = signal[i];
        err = matmulTT(delta, weights[i]);

        // Remove the error of bias and propagate backward
        err = err(span, seq(1, out.dims(1)));
    }
}

ann::ann(vector<int> layers, double range) :
    num_layers(layers.size()),
    weights(layers.size() - 1),
    weights_inc(layers.size() - 1)
{
    // Generate uniformly distributed random numbers between [-range/2,range/2]
    for (int i = 0; i < num_layers - 1; i++) {
        weights[i] = range * randu(layers[i] + 1, layers[i + 1]) - range/2;
        weights_inc[i] = constant(0.0, layers[i] + 1, layers[i + 1]);
    }
}

array ann::predict(const array &input)
{
    vector<array> signal = forward_propagate(input);
    array out = signal[num_layers - 1];
    return out;
}

float ann::train(const array &input, const array &target,
                  const array &input_test, const array &target_test,
                  float lr, float momentum, int max_epochs, int batch_size,
                  float maxerr, int testing_interval, bool verbose)
{

    const int num_samples = input.dims(0);
    std::cout << "Num samples: " << num_samples << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;
    
    double err = 0;
    
    // Training the entire network
    for (int i = 0; i < max_epochs; i++) {
        
        array idx = randu(batch_size, 1, u32);
        idx = mod(idx, num_samples);

        array x = lookup(input, idx, 0);
        array y = lookup(target, idx, 0);

        // Propagate the inputs forward
        vector<array> signals = forward_propagate(x);
        array out = signals[num_layers - 1];

        // Propagate the error backward
        back_propagate(signals, y, lr, momentum);
        
        if (i % testing_interval == 0) {
            // Testing
            array out = predict(input_test);
            err = error(out, target_test);
            if (verbose) {
                printf("Epoch: %4d, Error: %0.4f\n", i, err);
            }
        }
            
        // Check if convergence criteria has been met
        if (err < maxerr) {
            printf("Converged on Epoch: %4d\n", i + 1);
            return err;
        }
    }
    
    return err;
}

void load_data(array &train_images, array &train_target, array &test_images, array &test_target)
{
    const std::string test_labels_file = "t10k-labels.idx1-ubyte";
    const std::string test_features_file = "t10k-images.idx3-ubyte";
    const std::string train_labels_file = "train-labels.idx1-ubyte";
    const std::string train_features_file = "train-images.idx3-ubyte";
    
    std::vector<float> data;
    size_t samples;
    size_t features;
    size_t labels;

    read_mnist_images_file(train_features_file, data, samples, features); 
    //std::cout << samples << " " << features << " " << data.size() << std::endl;
    // saved in col-major order
    train_images = af::array(static_cast<int>(features), static_cast<int>(samples), &data[0]);
    
    read_mnist_labels_file(train_labels_file, data, samples, labels);
    //std::cout << samples << " " << labels << " " << data.size() << std::endl;
    // saved in col-major order
    train_target = af::array(static_cast<int>(labels), static_cast<int>(samples), &data[0]);
    
    read_mnist_images_file(test_features_file, data, samples, features); 
    //std::cout << samples << " " << features << " " << data.size() << std::endl;
    // saved in col-major order
    test_images = af::array(static_cast<int>(features), static_cast<int>(samples), &data[0]);
    
    read_mnist_labels_file(test_labels_file, data, samples, labels);
    
    // saved in col-major order
    test_target = af::array(static_cast<int>(labels), static_cast<int>(samples), &data[0]);
    //std::cout << samples << " " << labels << " " << data.size() << std::endl;
    
    train_images = train_images.T();
    train_target = train_target.T();
    test_images = test_images.T();
    test_target = test_target.T();
        
}

int ann_demo(bool console, int perc)
{
    printf("** ArrayFire ANN Demo **\n\n");

    array train_images, test_images;
    array train_target, test_target;
    int num_classes, num_train, num_test;

    // Load mnist data
    load_data(train_images, train_target, test_images, test_target);    

    // Reshape images into feature vectors
    //array train_feats = moddims(train_images, feature_size, num_train).T();
    //array test_feats  = moddims(test_images , feature_size, num_test ).T();

    //train_target = train_target.T();
    //test_target  = test_target.T();

    // Network parameters
    vector<int> layers;
    layers.push_back(train_images.dims(1));
    layers.push_back(65);
    layers.push_back(25);
    layers.push_back(train_target.dims(1));

    // Create network
    ann network(layers);

    // Train network
    timer::start();
    network.train(train_images, train_target,
                  test_images, test_target,
                  0.1,
                  0.9);
    af::sync();
    double train_time = timer::stop();

    // Run the trained network and test accuracy.
    array train_output = network.predict(train_images);
    array test_output  = network.predict(test_images);


    // Benchmark prediction
    af::sync();
    timer::start();
    for (int i = 0; i < 100; i++) {
        network.predict(test_images);
    }
    af::sync();
    double test_time = timer::stop() / 100;

    printf("\nTraining set:\n");
    printf("Accuracy on training data: %2.2f\n",
           accuracy(train_output, train_target));

    printf("\nTest set:\n");
    printf("Accuracy on testing  data: %2.2f\n",
           accuracy(test_output , test_target ));

    printf("\nTraining time: %4.4lf s\n", train_time);
    printf("Prediction time: %4.4lf s\n\n", test_time);

    if (!console) {
        // Get 20 random test images.
        test_output = test_output.T();
        //display_results<true>(test_images, test_output, test_target.T(), 20);
    }

    return 0;
}

int main(int argc, char** argv)
{
    int device   = argc > 1 ? atoi(argv[1]) : 0;
    bool console = argc > 2 ? argv[2][0] == '-' : false;
    int perc     = argc > 3 ? atoi(argv[3]) : 60;

    try {

        af::deviceset(device);
        af::info();
        return ann_demo(console, perc);

    } catch (af::exception &ae) {
        std::cout << ae.what() << std::endl;
    }

}