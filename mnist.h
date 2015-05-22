/* 
 * File:   mnist.h
 * Author: jordi
 *
 * Created on May 21, 2015, 3:58 PM
 */

#ifndef MNIST_H
#define	MNIST_H

void read_mnist_images_file(const std::string filename, 
                            std::vector<float> &v, 
                            size_t &r, 
                            size_t &c);

void read_mnist_labels_file(const std::string filename, 
                            std::vector<float> &v, 
                            size_t &r, 
                            size_t &c);

void print_mnist_image_txt(std::vector<float> &v, 
                           size_t offset, 
                           uint8_t rows, 
                           uint8_t cols);

void print_mnist_label_txt(std::vector<float> &v, 
                           size_t offset, 
                           uint8_t out);

#endif	/* MNIST_H */

