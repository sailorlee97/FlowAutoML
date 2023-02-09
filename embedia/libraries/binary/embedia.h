/* 
 * EmbedIA 
 * C LIBRARY FOR THE IMPLEMENTATION OF NEURAL NETWORKS ON MICROCONTROLLERS
 */
#ifndef _EMBEDIA_H
#define _EMBEDIA_H

#include <stdint.h>
#include <math.h>
{includes}






/* definition of global masks and data types */



#if tamano_del_bloque == 8
typedef uint8_t xBITS;
static const uint8_t mascara_global_bits[8] = { 128 , 64 , 32 , 16 , 8 , 4 , 2 , 1};
#elif tamano_del_bloque == 16
typedef uint16_t xBITS;
static const uint16_t mascara_global_bits[16] = { 32768 , 16384 , 8192 , 4096 , 2048 , 1024 , 512 , 256 , 128 , 64 , 32 , 16 , 8 , 4 , 2 , 1};
#elif tamano_del_bloque == 32
typedef uint32_t xBITS;
static const uint32_t mascara_global_bits[32] = { 2147483648 , 1073741824 , 536870912 , 268435456 , 134217728 , 67108864 , 33554432 , 16777216 , 8388608 , 4194304 , 2097152 , 1048576 , 524288 , 262144 , 131072 , 65536 , 32768 , 16384 , 8192 , 4096 , 2048 , 1024 , 512 , 256 , 128 , 64 , 32 , 16 , 8 , 4 , 2 , 1};
#elif tamano_del_bloque == 64
typedef uint64_t xBITS;
static const uint64_t mascara_global_bits[64] = { 9223372036854775808 , 4611686018427387904 , 2305843009213693952 , 1152921504606846976 , 576460752303423488 , 288230376151711744 , 144115188075855872 , 72057594037927936 , 36028797018963968 , 18014398509481984 , 9007199254740992 , 4503599627370496 , 2251799813685248 , 1125899906842624 , 562949953421312 , 281474976710656 , 140737488355328 , 70368744177664 , 35184372088832 , 17592186044416 , 8796093022208 , 4398046511104 , 2199023255552 , 1099511627776 , 549755813888 , 274877906944 , 137438953472 , 68719476736 , 34359738368 , 17179869184 , 8589934592 , 4294967296 , 2147483648 , 1073741824 , 536870912 , 268435456 , 134217728 , 67108864 , 33554432 , 16777216 , 8388608 , 4194304 , 2097152 , 1048576 , 524288 , 262144 , 131072 , 65536 , 32768 , 16384 , 8192 , 4096 , 2048 , 1024 , 512 , 256 , 128 , 64 , 32 , 16 , 8 , 4 , 2 , 1};
#else
typedef uint8_t xBITS;
static const uint8_t mascara_global_bits[8] = { 128 , 64 , 32 , 16 , 8 , 4 , 2 , 1};
#endif // tamano_del_bloque


/* STRUCTURE DEFINITION */


/*
 *Structure that stores the weights of a binary filter.
 *Specifies the number of channels (uint16_t channels), their size (uint16_t kernel_size), the weights (xBITS *bitarray;) and the bias (float bias).
 */

typedef struct{
    uint16_t channels;
    uint16_t kernel_size;
    const xBITS *bitarray;
    float bias;
}quant_filter_t;


/*
 * Structure that models a binary convolutional layer.
 * Specifies the number of filters (uint16_t n_filters) and a vector of quant filters (quant_filter_t * filters). 
 */

typedef struct{
    uint16_t n_filters;
    quant_filter_t * filters;
}quantconv2d_layer_t;


/*
 * Structure that models a binary neuron.
 * Specifies the weights of the neuron as a vector (xBITS  * weights) and the bias (float bias).
 */

typedef struct{
    const xBITS  * weights;
    float bias;
}quant_neuron_t;


/*
 * Structure that models a binary dense layer.
 * Specifies the number of neurons (uint16_t n_neurons) and a vector of quant neurons (quant_neuron_t  * neurons). 
 */

typedef struct{
    uint16_t n_neurons;
    quant_neuron_t * neurons;
}quantdense_layer_t;


/*
 * Structure that stores an array of float data (float * data) in vector form.
 * Specifies the number of channels, the width and the height of the array.
 */

typedef struct{
    uint16_t channels;
    uint16_t width;
    uint16_t height;
    float  * data;
}data3d_t;

typedef struct{
    uint16_t width;
    uint16_t height;
    float  * data;
}data2d_t;

typedef struct{
    uint32_t length;
    float  * data;
}data1d_t;


/*
 * Structure that stores the weights of a filter.
 Specifies the number of channels (uint16_t channels), their size (uint16_t kernel_size), the weights (float * weights) and the bias (float bias),
 * the weights (float * weights) and the bias (float bias).
 */
typedef struct{
    uint16_t channels;
    uint16_t kernel_size;
    const float  * weights;
    float  bias; 
}filter_t;

/*
 * Structure that models a convolutional layer.
 * Specifies the number of filters (uint16_t n_filters) and a vector of filters (filter_t * filters). 
 */
typedef struct{
    uint16_t n_filters;
    filter_t * filters; 
}conv2d_layer_t;

/*
 * Structure that models a separable layer...
 * Specifies the number of filters (uint16_t n_filters,
 * a filter of the specified size (filter_t depth_filter) 
 * a vector of 1x1 filters (filter_t * point_filters) 
 */
typedef struct{
    uint16_t n_filters;
    filter_t depth_filter;
    filter_t * point_filters; 
}separable_conv2d_layer_t;

/*
 * Structure that models a neuron.
 * Specifies the weights of the neuron as a vector (float * weights) and the bias (float bias).
 */
typedef struct{
    const float  * weights;
    float  bias;
}neuron_t;

/*
 * Structure that models a dense layer.
 * Specifies the number of neurons (uint16_t n_neurons) and a vector of neurons (neuron_t * neurons). 
 */
typedef struct{
    uint16_t n_neurons;
    neuron_t * neurons;
}dense_layer_t;


/*
 * Pooling Structure
 */
 
typedef struct{
    uint16_t size;
    uint16_t strides;
} pooling2d_layer_t;

/*********************************************************************************************************************************/

/* structure for normalization type (x_i-s_i)/d_i and (x_i)/d_i
 *  - standard normalization  : (x_i-mean_i)/std_dev_i
 *  - min_max normalization   : (x_i-min_i) / (max_i-min_i)
 *  - robust normalization    : (x_i-q1_i)  / (q3_i-q1_i)
 *  - abs_max_normalization   : (x_i)/(abs_max_xi)
 */

typedef struct{
    const float *sub_val;
    const float *inv_div_val;
} normalization_layer_t;


/* 
 * Structure for BatchNormalization layer.
 * Contains vectors for the four parameters used for normalization.
 * The number of each of the parameters is determined by the number of channels of the previous layer.
 */
typedef struct {
    uint32_t length;
    const float *beta;
    // const float *gamma;
    const float *moving_mean;
    const float *moving_inv_std_dev; // = gamma / sqrt(moving_variance + epsilon)
} batch_normalization_layer_t;


/* LIBRARY FUNCTIONS PROTOTYPES */


/* 
 * quantdense_layer()
 * Performs feed forward of a binary dense layer (quantdense_layer_t) on a given input data set.
 * Parameters:
 *   - dense_layer => structure with the binary weights of the neurons of the dense layer.  
 *   - input       => structure data1d_t with the input data to process. 
 *   - *output     => structure data1d_t to store the output result.
 */
void quantdense_layer(quantdense_layer_t dense_layer, data1d_t input, data1d_t * output);


/* 
 * quantconv2d_layer()
 * Función que se encarga de aplicar la convolución binaria de una capa de  * filtros (quantconv2d_layer_t)
 * sobre un determinado conjunto de datos de entrada.
 * Parámetros:
 *          quantconv2d_layer_t layer  =>  capa convolucional con filtros 
 *                                        cargados
 *         	      data3d_t input  =>  datos de entrada de tipo data3d_t
 *             data3d_t * output  =>  puntero a la estructura data3d_t donde se guardará el resultado
 */
void quantconv2d_layer(quantconv2d_layer_t layer,data3d_t input, data3d_t *output);



/* 
 * quantconv2d_input_not_binary_layer()
 * Función que se encarga de aplicar la convolución binaria (entrada no cuantizada) de una capa de filtros (quantconv2d_layer_t)
 * sobre un determinado conjunto de datos de entrada.
 * Parámetros:
 *          quantconv2d_layer_t layer  =>  capa convolucional con filtros 
 *                                        cargados
 *         	      data3d_t input  =>  datos de entrada de tipo data3d_t
 *             data3d_t * output  =>  puntero a la estructura data3d_t donde se guardará el resultado
 */
void quantconv2d_input_not_binary_layer(quantconv2d_layer_t layer,data3d_t input, data3d_t *output);


/* 
 * conv2d_layer()
 * Función que se encarga de aplicar la convolución de una capa de filtros (conv_layer_t)
 * sobre un determinado conjunto de datos de entrada.
 * Parámetros:
 *          conv_layer_t layer  =>  capa convolucional con filtros cargados
 *         	      data3d_t input  =>  datos de entrada de tipo data3d_t
 *             data3d_t * output  =>  puntero a la estructura data3d_t donde se guardará el resultado
 */
void conv2d_layer(conv2d_layer_t layer, data3d_t input, data3d_t * output);

/* 
 * separable_conv2d_layer()
 *  Function in charge of applying the convolution of a filter layer (conv_layer_t) on a given input data set.
 * 
 * Parameters:
 *   layer => convolutional layer with loaded filters.
 *   input => input data of type data3d_t
 *   *output => pointer to the data3d_t structure where the result will be saved.
 */
void separable_conv2d_layer(separable_conv2d_layer_t layer, data3d_t input, data3d_t * output);


/* 
 * dense_layer()
 * Performs feed forward of a dense layer (dense_layer_t) on a given input data set.
 * Parameters:
 *   dense_layer => structure with the weights of the neurons of the dense layer.  
 *   input       => structure data1d_t with the input data to process. 
 *   *output     => structure data1d_t to store the output result.
 */
void dense_layer(dense_layer_t dense_layer, data1d_t input, data1d_t * output);

/* 
 * max_pooling2d_layer()
 * Maxpooling layer, for now supports square size and stride. No support for padding 
 * Parameters:
 *   pool_size => size for pooling
 *   stride    => stride for pooling
 *   input     => input data 
 *   output    => output data
 */
void max_pooling2d_layer(pooling2d_layer_t pool, data3d_t input, data3d_t* output);

/* 
 * avg_pooling_2d()
 * Function that applies an average pooling to an input with a window size of received 
 * by parameter (uint16_t strides)
 *
 * Parameters:
 *   input => input data of type data3d_t.
 *   *output => pointer to the data3d_t structure where the result will be stored.
 */
void avg_pooling2d_layer(pooling2d_layer_t pool, data3d_t input, data3d_t* output);


/* 
 * flatten3d_layer()
 * Performs a variable shape change. 
 * Converts the data format from data3d_t array format to data1d_t vector.
 * (prepares data for input into a layer of type dense_layer_t).
 * Parameters:
 *    input => input data of type data3d_t.
 *    *output => pointer to the data1d_t structure where the result will be stored.
 */
 void flatten3d_layer(data3d_t input, data1d_t * output);
     
/* 
 * argmax()
 * Finds the index of the largest value within a vector of data (data1d_t)
 * Parameters:
 *   data => data of type data1d_t to search for max.
 *
 * Returns:
 *  search result - index of the maximum value
 */
uint16_t argmax(data1d_t data);


/***************************************************************************************************************************/
/* Activation functions/layers */

void softmax_activation(float *data, uint32_t length);

void relu_activation(float *data, uint32_t length);

void leakyrelu_activation(float *data, uint32_t length, float alpha);

void tanh_activation(float *data, uint32_t length);

void sigmoid_activation(float *data, uint32_t length);

void softsign_activation(float *data, uint32_t length);



/***************************************************************************************************************************/
/* Normalization layers */

/* Normalization function for:
 *  standard normalization  : (x_i-mean_i)/std_dev_i
 *  min_max normalization   : (x_i-min_i) / (max_i-min_i)
 *  robust normalization    : (x_i-q2_i)  / (q3_i-q1_i)
 */
void normalization1(normalization_layer_t s, data1d_t input, data1d_t * output);

#define standard_norm_layer(norm, input, output) normalization1(norm, input, output)

#define min_max_norm_layer(norm, input, output) normalization1(norm, input, output)

#define robust_norm_layer(norm, input, output) normalization1(norm, input, output)


/* Normalization function for:
 *  abs_max_normalization   : (x_i)/(abs_max_xi)
 */
void normalization2(normalization_layer_t s, data1d_t input, data1d_t * output);

#define max_abs_norm_layer(norm, input, output) normalization2(norm, input, output)


void batch_normalization_layer(batch_normalization_layer_t norm, uint32_t length, float *data);


void batch_normalization3d_layer(batch_normalization_layer_t layer, data3d_t *data);

void batch_normalization1d_layer(batch_normalization_layer_t layer, data1d_t *data);

/* functions */
/*
* sign function, used to binarize the input
*/
static inline uint8_t sign(float x);


/*
* Brian Kernighan's algorithm, is used to count high bits (logical 1) 
* efficiently.
*/
static inline int count_set_bits_Brian_Kernighan_algorithm(xBITS n);


/*
* POPCOUNT function 
*/
static inline float POPCOUNT(xBITS n);


/*
* applies the XNOR function efficiently, between two numbers loaded in 
* registers
*/
static inline xBITS XNOR(register xBITS a,register xBITS b);

/* Tranformation Layers
 *
 */

/* image_adapt_layer()
 *  Converts Tensorflow/Keras Image (Height, Width, Channel) to Embedia format (Channel, Height, Width).
 *  Usually required for first convolutional layer
 * Parameters:
 *  input   => input data of type data3d_t.
 *  *output => pointer to the data3d_t structure where the result will be stored.
 */
void image_adapt_layer(data3d_t input, data3d_t * output);


#endif