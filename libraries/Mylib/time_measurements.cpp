#include "time_measurements.h"
#include <cstdio>


unsigned long InvokeSubgraph_time;
unsigned long InvokeSubgraph_num;
unsigned long InvokeSubgraph_NumSubgraphOperators_time;
unsigned long InvokeSubgraph_NumSubgraphOperators_num;
unsigned long InvokeSubgraph_for_time;
unsigned long InvokeSubgraph_for_num;
unsigned long InvokeSubgraph_declare_node_time;
unsigned long InvokeSubgraph_declare_node_num;
unsigned long InvokeSubgraph_declare_registration_time;
unsigned long InvokeSubgraph_declare_registration_num;
unsigned long InvokeSubgraph_error_priflier_time;
unsigned long InvokeSubgraph_error_priflier_num;
unsigned long InvokeSubgraph_operation_time[8];
unsigned long InvokeSubgraph_operation_num;
unsigned long InvokeSubgraph_ResetTempAllocations_time;
unsigned long InvokeSubgraph_ResetTempAllocations_num;
unsigned long InvokeSubgraph_kTfLiteError_time;
unsigned long InvokeSubgraph_kTfLiteError_num;

unsigned long s2;
unsigned long t2;


unsigned long s[10];
unsigned long t[10];
unsigned long n[10];

unsigned long test_start;
unsigned long test_end;
unsigned long test_num;

//초기화
void reset_measurements() {
  InvokeSubgraph_time = 0;
  InvokeSubgraph_num = 0;
  InvokeSubgraph_NumSubgraphOperators_time = 0;
  InvokeSubgraph_NumSubgraphOperators_num = 0;
  InvokeSubgraph_for_time = 0;
  InvokeSubgraph_for_num = 0;
  InvokeSubgraph_declare_node_time = 0;
  InvokeSubgraph_declare_node_num = 0;
  InvokeSubgraph_declare_registration_time = 0;
  InvokeSubgraph_declare_registration_num = 0;
  InvokeSubgraph_error_priflier_time = 0;
  InvokeSubgraph_error_priflier_num = 0;
  InvokeSubgraph_operation_time[8] = {0};
  InvokeSubgraph_operation_num = 0;
  InvokeSubgraph_ResetTempAllocations_time = 0;
  InvokeSubgraph_ResetTempAllocations_num = 0;
  InvokeSubgraph_kTfLiteError_time = 0;
  InvokeSubgraph_kTfLiteError_num = 0;



  s[10] = {0};
  t[10] = {0};
  n[10] = {0};
  s2 = 0;
  t2 = 0;

  test_start=0;
  test_end=0;
  test_num=0;

}

// 전처리 함수: float32를 signed int8로 변환하고 시리얼 모니터에 출력
void preprocess(const float* input_data, int8_t* quantized_data, int size, float scale, int zero_point) {
  for (int i = 0; i < size; ++i) {
    // float32 값을 signed int8로 변환
    int quantized_value = static_cast<int>(input_data[i] / scale + zero_point);
    // signed int8 범위로 클램핑
    if (quantized_value < -128) quantized_value = -128;
    if (quantized_value > 127) quantized_value = 127;
    quantized_data[i] = static_cast<int8_t>(quantized_value);
  }

}

//사용 안함
// 후처리 함수: signed int8를 float32로 변환
void postprocess(const int8_t* quantized_data, float* output_data, int size, float scale, int zero_point) {
  for (int i = 0; i < size; ++i) {
    output_data[i] = (quantized_data[i] - zero_point) * scale;
  }
}