#ifndef TIME_MEASUREMENTS_H
#define TIME_MEASUREMENTS_H


#include <Arduino.h>


// 여러 함수의 수행 시간을 저장할 전역 변수 선언
//micro_interpreter.cpp
extern unsigned long InvokeSubgraph_time;
extern unsigned long InvokeSubgraph_num;

extern unsigned long InvokeSubgraph_NumSubgraphOperators_time;
extern unsigned long InvokeSubgraph_NumSubgraphOperators_num;

extern unsigned long InvokeSubgraph_for_time;
extern unsigned long InvokeSubgraph_for_num;

extern unsigned long InvokeSubgraph_declare_node_time;
extern unsigned long InvokeSubgraph_declare_node_num;

extern unsigned long InvokeSubgraph_declare_registration_time;
extern unsigned long InvokeSubgraph_declare_registration_num;

extern unsigned long InvokeSubgraph_error_priflier_time;
extern unsigned long InvokeSubgraph_error_priflier_num;

extern unsigned long InvokeSubgraph_operation_time[8];
extern unsigned long InvokeSubgraph_operation_num;

extern unsigned long InvokeSubgraph_ResetTempAllocations_time;
extern unsigned long InvokeSubgraph_ResetTempAllocations_num;

extern unsigned long InvokeSubgraph_kTfLiteError_time;
extern unsigned long InvokeSubgraph_kTfLiteError_num;

extern unsigned long s2;
extern unsigned long t2;

extern unsigned long s[10];
extern unsigned long t[10];
extern unsigned long n[10];


//test
extern unsigned long test_start;
extern unsigned long test_end;
extern unsigned long test_num;

void reset_measurements();
void preprocess(const float* , int8_t* , int , float , int );
void postprocess(const int8_t* , float* , int , float , int );






#endif // TIME_MEASUREMENTS_H

