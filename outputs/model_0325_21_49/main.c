#include <stdio.h>
#include "embedia.h"
#include "model_0325_21_49_model.h"
#include "example_file.h"


 // esto no iría aca, solo sería para arduino


data3d_t input = { INPUT_CHANNELS, INPUT_WIDTH, INPUT_HEIGHT,  NULL};


data1d_t results;



int main(void){
	
    
    // sample intitialization
    input.data = sample_data;

    // model initialization
    model_init();


    // make model prediction
    // uncomment corresponding code

    int prediction = model_predict_class(input, &results);

    // print predicted class id
    printf("Prediction class id: %d\n", prediction);

    printf("   Example class id: %d\n", sample_data_id);

    /*

    model_predict(input, &results);
    printf("prediccion: %5f", results.data[0]);
    */



	return 0;
}