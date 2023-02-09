#ifndef EMBEDIA_DEBUG_H_INCLUDED
#define EMBEDIA_DEBUG_H_INCLUDED

#include "embedia.h"

// exporter must define EMBEDIA_DEBUG macro
// 0 => NO DEBUG, 1 => DATA HEADER, 2 => DATA CONTENT
{EMBEDIA_DEBUG}




void print_data1d_t(const char *head_text, data1d_t data);

void print_data2d_t(const char *head_text, data2d_t data);

void print_data3d_t(const char *head_text, data3d_t data);

/*
 * print_filter_t()
 * Imprime los valores de los pesos del filtro y sus dimensiones
 * Parámetros:
 *                filter_t filtro => filtro a imprimir
 */

void print_filter_t(const char *head_text, filter_t filter);


#endif // EMBEDIA_DEBUG_H_INCLUDED
