#ifndef FIXED_H
#define FIXED_H

#include <stdint.h>

typedef int32_t fixed;
typedef int64_t dfixed;

#define FIX_SIZE 32
#define FIX_FRC_SZ 18
#define FIX_INT_SZ (FIX_SIZE - FIX_FRC_SZ)
#define FIX_FRC_MSK  (((fixed)1 << FIX_FRC_SZ) - 1)

//////////////////////////////////// Constantes ////////////////////////////////////

#define FIX_HALF (FIX_ONE >> 1)
#define FIX_ZERO 0
#define FIX_ONE ((fixed)((fixed)1 << FIX_FRC_SZ))
#define FIX_TWO (FIX_ONE + FIX_ONE)
#define FIX_E  FLOAT_TO_FIXED(2.7182818284590452354)
#define FIX_PI FLOAT_TO_FIXED(3.14159265358979323846)
#define FIX_MAX (fixed)(((dfixed)1 << (FIX_SIZE-1)) -1)
#define FIX_MIN (-FIX_MAX)

//////////////////////////////////// Macros de conversion ////////////////////////////////////

#define FIXED_TO_DOUBLE(F) ((double) ((F)*((double)(1)/(double)(1L << FIX_FRC_SZ))))
#define FIXED_TO_FLOAT(F) ((float) ((F)*((float)(1)/(float)(1L << FIX_FRC_SZ))))
#define FIXED_TO_INT(F) ((fixed)(F) >> FIX_FRC_SZ)
#define FIXED_FRAC(F) ( (fixed)(F) & FIX_FRC_MSK )
#define FIXED_INT(F) ( (fixed)(F) & ~FIX_FRC_MSK )
#define FLOAT_TO_FIXED(F) ((fixed)((F) * FIX_ONE + ((F) >= 0 ? 0.5 : -0.5)))
#define INT_TO_FIXED(I) ((fixed)(I) << FIX_FRC_SZ)

#define FL2FX(F) FLOAT_TO_FIXED(F)
#define FX2FL(F) FIXED_TO_FLOAT(F)



//////////////////////////////////// Macros aritmeticas ////////////////////////////////////

#define FIXED_ADD(A,B) ((A) + (B))

#define FIXED_SUB(A,B) ((A) - (B))

#define FIXED_MUL(A,B)            \
  ((fixed)(((dfixed)(A) * (dfixed)(B)) >> FIX_FRC_SZ))

#define FIXED_DIV(A,B)           \
  ((fixed)(((dfixed)(A) << FIX_FRC_SZ) / (dfixed)(B)))


//////////////////////////////////// Macros adicionales ////////////////////////////////////

#define FIXED_ABS(A) ((A) < 0 ? -(A) : (A))
#define FIXED_CEIL(A) ( FIXED_INT(A) +  (FIXED_FRAC(A) ? FIX_ONE : 0) )
#define FIXED_FLOOR(A) ( FIXED_INT(A) )

//////////////////////////////////// funciones de conversion de tipos ////////////////////////////////////

    // Devuelve un numero en punto fijo a partir de uno en punto flotante
    fixed float_to_fixed(float f);

    // Devuelve un numero en punto fijo a partir de un entero
    fixed int_to_fixed(int32_t i);

    // Devuelve la representacion en punto flotante de doble precision de un numero en punto fijo
    double fixed_to_double(fixed f);

    // Devuelve la representacion en punto flotante de un numero en punto fijo
    float fixed_to_float(fixed f);

    // Devuelve el entero a partir de un numero en punto fijo
    int32_t fixed_to_int(fixed f);


//////////////////////////////////// funciones aritmeticas ////////////////////////////////////

    // Devuelve la suma en punto fijo entre a y b
    fixed fixed_add(fixed a, fixed b);

    // Devuelve la resta en punto fijo entre a y b
    fixed fixed_sub(fixed a, fixed b);

    // Devuelve la multiplicacion en punto fijo entre a y b
    fixed fixed_mul(fixed a, fixed b);

    // Devuelve la division en punto fijo entre a y b
    fixed fixed_div(fixed a, fixed b);


//////////////////////////////////// funciones especiales ////////////////////////////////////


    // Devuelve la raiz cuadrada del numero a o -1 en caso de error
    fixed fixed_sqrt(fixed a);

    // Devuelve el valor de exp(a)
    fixed fixed_exp(fixed a);

    // Devuelve x * 2^exp
    fixed fixed_ldexp(fixed x, int exp);

    // Devuelve el logaritmo de x
    fixed fixed_log(fixed x);

    // Devuelve el logaritmo de base b del valor x
    fixed fixed_logn(fixed x, fixed b);

    // Devuelve n^exp
    fixed fixed_pow(fixed n, fixed exp);

    // Devuelve tanh(x)
    fixed fixed_tanh(fixed x);


//////////////////////////////////// funciones adicionales ////////////////////////////////////

    // Devuelve el valor absoluto de a
    fixed fixed_abs(fixed a);

    // Devuelve el entero menor x tal que  x >= a
    fixed fixed_ceil(fixed a);

    // Devuelve el entero mayor x tal que  x <= a
    fixed fixed_floor(fixed a);


#endif
