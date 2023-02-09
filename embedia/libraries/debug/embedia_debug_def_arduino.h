#include <Arduino.h>

#ifdef FIX_SIZE
#define DBG_FL(x) FX2FL(x)
#else
#define DBG_FL(x) x
#endif // FIX_SIZE

#define DBG_INT_PART 4
#define DBG_FRC_PART 6
#define DBG_FLOAT_SIZE (DBG_INT_PART+DBG_FRC_PART+1)

#if EMBEDIA_DEBUG > 0
static char temp_str[DBG_FLOAT_SIZE+1];
#endif

#if defined (__AVR__) || (__avr__) || (ARDUINO_ARCH_AVR)
    // sprintf of many AVR doesn't support float numbers
    #define PRINT_FL(txt, fl)                                     \
        Serial.print(txt);                                        \
        dtostrf(DBG_FL(fl),DBG_FLOAT_SIZE,DBG_FRC_PART,temp_str); \
        Serial.print(temp_str)

    #define PRINT_FL_LN(txt, fl)                                  \
        Serial.print(txt);                                        \
        dtostrf(DBG_FL(fl),DBG_FLOAT_SIZE,DBG_FRC_PART,temp_str); \
        Serial.println(temp_str)
#else

   #define DBG_FLOAT_FMT "%11.6f"

   #define PRINT_FL(txt, fl)                        \
      Serial.print(txt);                            \
      sprintf(temp_str, DBG_FLOAT_FMT, DBG_FL(fl)); \
      Serial.print(temp_str)

   #define PRINT_FL_LN(txt, fl)                     \
      Serial.print(txt);                            \
      sprintf(temp_str, DBG_FLOAT_FMT, DBG_FL(fl)); \
      Serial.println(temp_str)

#endif // AVR


#define PRINT_INT(txt, n)           \
    Serial.print(txt);              \
    Serial.print(n)

#define PRINT_INT_LN(txt, n)        \
    Serial.print(txt);              \
    Serial.println(n)

#define PRINT_TXT_LN(txt)           \
    Serial.println(txt)
