/*
 * EmbedIA
 * C LIBRARY FOR THE IMPLEMENTATION OF NEURAL NETWORKS ON MICROCONTROLLERS
 */

#include "embedia.h"


typedef struct{
    size_t  size;
    void  * data;
} raw_buffer;


raw_buffer buffer1 = {0, NULL};
raw_buffer buffer2 = {0, NULL};

void * swap_alloc(size_t s){
    static raw_buffer * last_buff = &buffer2;
    last_buff = (last_buff==&buffer1) ? &buffer2 : &buffer1;
    
    if (last_buff->size < s){
        last_buff->data = realloc(last_buff->data, s);
        last_buff->size = s;
    }

    return last_buff->data;
}

/*
 * conv2d()
 *  Function that performs convolution between a filter and a data set.
 * Parameters:
 *  filter => filter structure with weights loaded.
 *  input => input data of type data3d_t
 *  *output => pointer to the data3d_t structure where the result will be stored
 *  delta => positioning of feature_map inside output.data
 */

void conv2d(filter_t filter, data3d_t input, data3d_t * output, uint32_t delta){
    uint32_t i,j,k,l,c;
    dfixed suma;

	for (i=0; i<output->height; i++){
		for (j=0; j<output->width; j++){
			suma = 0;
			for (c=0; c<filter.channels; c++){
				for (k=0; k<filter.kernel_size; k++){
					for (l=0; l<filter.kernel_size; l++){
						suma += DFIXED_MUL(filter.weights[(c*filter.kernel_size*filter.kernel_size)+k*filter.kernel_size+l] , input.data[(c*input.height*input.width)+(i+k)*input.width+(j+l)]);
					}
				}
			}

			suma = suma + FIXED_TO_DFIXED(filter.bias);
			if (suma > DFIX_MAX)
				suma = FIX_MAX;
			else if (suma < DFIX_MIN)
				suma = FIX_MIN;
			else suma = DFIXED_TO_FIXED(suma);
			
			output->data[delta + i*output->width + j] = suma;
		}
	}

}

/*
 * conv2d_layer()
 *  Function in charge of applying the convolution of a filter layer (conv_layer_t) on a given input data set.
 * Parameters:
 *  layer => convolutional layer with loaded filters.
 *  input => input data of type data3d_t
 *  *output => pointer to the data3d_t structure where the result will be saved.
 */
 void conv2d_layer(conv2d_layer_t layer, data3d_t input, data3d_t * output){
    uint32_t delta;

    output->channels = layer.n_filters; //cantidad de filtros
    output->height   = input.height - layer.filters[0].kernel_size + 1;
    output->width    = input.width - layer.filters[0].kernel_size + 1;
    output->data     = (fixed*)swap_alloc( sizeof(fixed)*output->channels*output->height*output->width );

    for(uint32_t i=0; i<layer.n_filters; i++){
        delta = i*(output->height)*(output->width);
        conv2d(layer.filters[i],input,output,delta);
    }

}

static void depthwise(filter_t filter, data3d_t input, data3d_t * output){
    uint32_t i,j,k,l,c;
    dfixed suma;

	for (i=0; i<output->height; i++){
		for (j=0; j<output->width; j++){
			for (c=0; c<filter.channels; c++){
				suma=0;
				for (k=0; k<filter.kernel_size; k++){
					for (l=0; l<filter.kernel_size; l++){
						suma += DFIXED_MUL(filter.weights[(c*filter.kernel_size*filter.kernel_size)+k*filter.kernel_size+l] , input.data[(c*input.height*input.width)+(i+k)*input.width+(j+l)]);
					}
				}

				if (suma > DFIX_MAX)
					suma = FIX_MAX;
				else if (suma < DFIX_MIN)
					suma = FIX_MIN;
				else suma = DFIXED_TO_FIXED(suma);

				output->data[c*output->width*output->height + i*output->width + j] = suma;
			}
		}
	}
}

static void pointwise(filter_t filter, data3d_t input, data3d_t * output, uint32_t delta){
    uint32_t i,j,c;
    dfixed suma;

    for (i=0; i<output->height; i++){
        for (j=0; j<output->width; j++){
            suma = 0;
            for (c=0; c<filter.channels; c++){
                suma += DFIXED_MUL(filter.weights[c], input.data[(c*input.height*input.width)+i*input.width+j]);
            }

			suma = suma + FIXED_TO_DFIXED(filter.bias);
			if (suma > DFIX_MAX)
				suma = FIX_MAX;
			else if (suma < DFIX_MIN)
				suma = FIX_MIN;
			else suma = DFIXED_TO_FIXED(suma);
			
			output->data[delta + i*output->width + j] = suma;
        }
    }
}

/* 
 * separable_conv2d_layer()
 *  Function in charge of applying the convolution of a filter layer (conv_layer_t) on a given input data set.
 * Parameters:
 *  layer => convolutional layer with loaded filters.
 *  input => input data of type data3d_t
 *  *output => pointer to the data3d_t structure where the result will be saved.
 */

void separable_conv2d_layer(separable_conv2d_layer_t layer, data3d_t input, data3d_t * output){
    uint32_t delta;
    data3d_t depth_output;

    depth_output.channels = input.channels; //cantidad de canales
    depth_output.height   = input.height - layer.depth_filter.kernel_size + 1;
    depth_output.width    = input.width - layer.depth_filter.kernel_size + 1;
    depth_output.data     = (fixed*)swap_alloc( sizeof(fixed)*depth_output.channels*depth_output.height*depth_output.width );

    depthwise(layer.depth_filter,input,&depth_output);

    output->channels = layer.n_filters; //cantidad de filtros
    output->height   = depth_output.height;
    output->width    = depth_output.width;
    output->data     = (fixed*)swap_alloc( sizeof(fixed)*output->channels*output->height*output->width );
    
    for(uint32_t i=0; i<layer.n_filters; i++){
        delta = i*(output->height)*(output->width);
        pointwise(layer.point_filters[i],depth_output,output,delta);
    }
}

/*
 * neuron_forward()
 *  Function that performs the forward of a neuron in front of a given set of input data.
 * Parameters:
 *  neuron_t neuron => neuron with its weights and bias loaded.
 *  flatten_data_t input => input data in vector form (flatten_data_t).
 * Returns:
 *  fixed => result of the operation         
 */

static fixed neuron_forward(neuron_t neuron, data1d_t input){
    uint32_t i;
    dfixed result = 0;

    for(i=0;i<input.length;i++){
        result += DFIXED_MUL(input.data[i],neuron.weights[i]);
    }

	result =  result + FIXED_TO_DFIXED(neuron.bias);

	if (result > DFIX_MAX)
		return FIX_MAX;
	else if (result < DFIX_MIN)
		return FIX_MIN;
	return DFIXED_TO_FIXED(result);
}

/*
 * dense_layer()
 *  Performs feed forward of a dense layer (dense_layer_t) on a given input data set.
 * Parameters:
 *  dense_layer => structure with the weights of the neurons of the dense layer.
 *  input       => structure data1d_t with the input data to process.
 *  *output     => structure data1d_t to store the output result.
 */
void dense_layer(dense_layer_t dense_layer, data1d_t input, data1d_t * output){
    uint32_t i;

    output->length = dense_layer.n_neurons;
    output->data = (fixed*)swap_alloc(sizeof(fixed)*dense_layer.n_neurons);

    for(i=0;i<dense_layer.n_neurons;i++){
        output->data[i] = neuron_forward(dense_layer.neurons[i],input);
    }
}


/*
 * max_pooling2d_layer()
 *  Maxpooling layer, for now supports square size and stride. No support for padding
 * Parameters:
 *  pool_size => size for pooling
 *  stride    => stride for pooling
 *  input     => input data
 *  output    => output data
 */

void max_pooling2d_layer(pooling2d_layer_t pool, data3d_t input, data3d_t* output){
    uint32_t c,i,j,aux1,aux2;
    fixed max = -FIX_MAX;
    fixed num;

    // output->height = (input.height)/strides ;//+ (input.height%strides ? 0 : 1);
    // output->width =  (input.width )/strides ;//+ (input.width %strides ? 0 : 1);
    output->height = ((uint32_t) ((input.height - pool.size)/pool.strides)) + 1;
    output->width  = ((uint32_t) ((input.width - pool.size)/pool.strides)) + 1;
    output->channels = input.channels;
    output->data = (fixed*)swap_alloc(sizeof(fixed)*(output->channels)*(output->height)*(output->width));

    for (c=0; c<output->channels; c++){
        for (i=0; i<output->height; i++){
            for (j=0; j<output->width; j++){

                max = -FIX_MAX;

                for(aux1=0; aux1<pool.size; aux1++){
                        for(aux2=0; aux2<pool.size; aux2++){
                        
                        num = input.data[c*input.width*input.height + (i*pool.strides + aux1)*input.width + j*pool.strides + aux2];
                        
                        if(num>max){
                            max = num;
                        }
                    }
                }

                output->data[c*output->width*output->height + i*output->width + j] = max;
            }
        }    
    }
}


/*
 * avg_pooling_2d()
 *  Function that applies an average pooling to an input with a window size of received
 *  by parameter (uint16_t strides)
 * Parameters:
 *  input => input data of type data3d_t.
 *  *output => pointer to the data3d_t structure where the result will be stored.
 */

void avg_pooling2d_layer(pooling2d_layer_t pool, data3d_t input, data3d_t* output){
    uint32_t c,i,j,aux1,aux2;
    dfixed cant = INT_TO_FIXED(pool.size*pool.size);
    dfixed avg = 0;
    fixed num;

    // output->height = (input.height)/strides ;
    // output->width =  (input.width )/strides ;
    output->height = ((uint32_t) ((input.height - pool.size)/pool.strides)) + 1;
    output->width  = ((uint32_t) ((input.width - pool.size)/pool.strides)) + 1;
    output->channels = input.channels;
    output->data = (fixed*)swap_alloc(sizeof(fixed)*(output->channels)*(output->height)*(output->width));

    for (c=0; c<output->channels; c++){
        for (i=0; i<output->height; i++){
            for (j=0; j<output->width; j++){

                avg = 0;

                for(aux1=0; aux1<pool.size; aux1++){
                    for(aux2=0; aux2<pool.size; aux2++){
                        num = input.data[c*input.width*input.height + (i*pool.strides + aux1)*input.width + j*pool.strides + aux2];
                        avg += num;
                    }                  
                }

                output->data[c*output->width*output->height + i*output->width + j] = FIXED_DIV(avg,cant);
            }
        }    
    }
}

/*
 * softmax activation function
 * Parameters:
 *  *data  => array of values to update
 *  length => numbers of values to update
 */
void softmax_activation(fixed *data, uint32_t length){
    fixed m = -FIX_MAX;
    for (size_t i = 0; i < length; i++) {
        if (data[i] > m) {
            m = data[i];
        }
    }

    fixed sum = FL2FX(0.0);
    for (size_t i = 0; i < length; i++) {
        sum += fixed_exp(data[i] - m);
    }

    fixed offset = m + fixed_log(sum);
    for (size_t i = 0; i < length; i++) {
        data[i] = fixed_exp(data[i] - offset);
    }
}


/*
 * relu activation function
 * Parameters:
 *  *data  => array of values to update
 *  length => numbers of values to update
 */
void relu_activation(fixed *data, uint32_t length){
    uint32_t i;

    for (i=0;i<(length);i++){
        data[i] = data[i] < 0 ? 0 : data[i];
    }
}

/*
 * leaky relu activation function
 * Parameters:
 *  alfa   => coeficient to multiply negative values
 *  *data  => array of values to update
 *  length => numbers of values to update
 */
void leakyrelu_activation(fixed *data, uint32_t length, fixed alpha){
    uint32_t i;

    for (i=0;i<(length);i++){
        data[i] = data[i] < 0 ? FIXED_MUL(alpha, data[i]) : data[i];
    }
}


/*
 * tanh activation function: (2 / (1+e^(-2x)) -1
 * Parameters:
 *  *data  => array of values to update
 *  length => numbers of values to update
 */
void tanh_activation(fixed *data, uint32_t length){
    uint32_t i;

    for (i=0;i<length;i++){
        data[i] = fixed_tanh(data[i]);
        //data[i] = 2/(1+fixed_exp(-2*data[i])) - 1;
    }
}

/*
 * sigmoid activation function: 1 / (1 + exp(-x))
 * Parameters:
 *  *data  => array of values to update
 *  length => numbers of values to update
 */
void sigmoid_activation(fixed *data, uint32_t length){
    uint32_t i;

    for (i=0;i<length;i++){
        data[i] = 1 / (1 + fixed_exp(-data[i]));
    }
}

/*
 * softsign activation function: x / (|x| + 1)
 * Parameters:
 *  *data  => array of values to update
 *  length => numbers of values to update
 */
void softsign_activation(fixed *data, uint32_t length){
    uint32_t i;

    for (i=0;i<length;i++){
        data[i] = FIXED_DIV(data[i],(fixed_abs(data[i])+1));
    }
}

/*
 * softplus activation function: log(e^x + 1)
 * Parameters:
 *  *data  => array of values to update
 *  length => numbers of values to update
 */
void softplus_activation(fixed *data, uint32_t length){
    uint32_t i;

    for (i=0;i<length;i++){
        data[i] = fixed_log( fixed_exp(data[i])+1 );
    }
}


/*
 * flatten3d_layer()
 *  Performs a variable shape change.
 *  Converts the data format from data3d_t array format to data1d_t vector.
 *  (prepares data for input into a layer of type dense_layer_t).
 * Parameters:
 *  input => input data of type data3d_t.
 *  *output => pointer to the data1d_t structure where the result will be stored.
 */
void flatten3d_layer(data3d_t input, data1d_t * output){
    uint32_t c,i,j;
    uint32_t cantidad = 0;

    output->length = input.channels * input.height * input.width;
    output->data = (fixed*)swap_alloc(sizeof(fixed)*output->length);

    for(i=0;i<input.height;i++){
        for(j=0;j<input.width;j++){
            for(c=0;c<input.channels;c++){
                output->data[cantidad++] = input.data[(c*input.width*input.height)+(i*input.width)+j];
            }
        }
    }
}


/*
 * argmax()
 *  Finds the index of the largest value within a vector of data (data1d_t)
 * Parameters:
 *  data => data of type data1d_t to search for max.
 * Returns:
 *  search result - index of the maximum value
 */

uint32_t argmax(data1d_t data){
    fixed max = data.data[0];
    uint32_t pos = 0;

    for(uint32_t i=1;i<data.length;i++){
        if(data.data[i]>max){
            max = data.data[i];
            pos = i;
        }
    }

    return pos;
}

/*********************************************************************************************************************************/
/* Normalization layers */


void normalization1(normalization_layer_t n, data1d_t input, data1d_t * output){

    uint32_t i;

    output->length = input.length;
    output->data = (fixed*)swap_alloc(sizeof(fixed)*output->length);

    for(i=0; i<input.length; i++){
        output->data[i] =  FIXED_MUL((input.data[i]-n.sub_val[i]),n.inv_div_val[i]);
    }
}

void normalization2(normalization_layer_t n, data1d_t input, data1d_t * output){

    uint32_t i;

    output->length = input.length;
    output->data = (fixed*)swap_alloc(sizeof(fixed)*output->length);

    for(i=0; i<input.length; i++){
        output->data[i] = FIXED_MUL(input.data[i],n.inv_div_val[i]);
    }
}

/*
 * batch_normalization{X}d_layer()
 * Keras Batch Normalization
 * Parameters:
 *      batch_normlization_t norm =>  structure with batch normalization layer parameters
 *      *data  =>   pointer to data{X}d_t
 */

void batch_normalization1d_layer(batch_normalization_layer_t layer, data1d_t *data) {
    uint32_t i;
	dfixed d_data;

	for (i = 0; i < data->length; i++) {
		
		d_data = DFIXED_MUL((data->data[i] - layer.moving_mean[i]),layer.moving_inv_std_dev[i] + layer.beta[i]);

		if (d_data > DFIX_MAX)
			d_data = FIX_MAX;
		else if (d_data < DFIX_MIN)
			d_data = FIX_MIN;
		else 
			d_data = DFIXED_TO_FIXED(d_data);

		data->data[i] = d_data;
	}
}

void batch_normalization3d_layer(batch_normalization_layer_t layer, data3d_t *data) {
    uint32_t i, j, ilen = 0;
    uint32_t length = data->height * data->width;
	dfixed d_data;

    for (i = 0; i < data->channels; i++, ilen += length) {
        for (j = 0; j < length; j++) {
            d_data = DFIXED_MUL((data->data[ilen+j] - layer.moving_mean[i]),layer.moving_inv_std_dev[i] + layer.beta[i]);
			
			if (d_data > DFIX_MAX)
				d_data = FIX_MAX;
			else if (d_data < DFIX_MIN)
				d_data = FIX_MIN;
			else 
				d_data = DFIXED_TO_FIXED(d_data);

			data->data[ilen+j] = d_data;
		}
    }
}


/* image_adapt_layer()
 *  Converts Tensorflow/Keras Image (Height, Width, Channel) to Embedia format (Channel, Height, Width).
 *  Usually required for first convolutional layer
 * Parameters:
 *  input   => input data of type data3d_t.
 *  *output => pointer to the data3d_t structure where the result will be stored.
 */
void image_adapt_layer(data3d_t input, data3d_t * output){

    uint32_t i, j, c, l;

    output->channels = input.channels;
    output->height   = input.height;
    output->width    = input.width;
    output->data     = (fixed*)swap_alloc( sizeof(fixed)*output->channels*output->height*output->width );

    for (c=0, l=0; c < input.channels; c++){
        for (i=0; i < input.height; i++) {
            for (j=0; j < input.width; j++, l++ ){
                output->data[l] = input.data[i*input.channels*input.width+input.channels*j+c];
            }
        }
    }
 }