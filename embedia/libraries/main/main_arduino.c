{includes}

{input_data}

{output_data}

void setup(){{

    // Serial inicialization
    Serial.begin({baud_rate});

    // Model initialization
    model_init();

}}

void loop(){{

    {main_code}

    delay(5000);

}}