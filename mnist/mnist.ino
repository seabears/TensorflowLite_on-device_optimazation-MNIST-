
//#include "test.h" //테스트 이미지 배열(5) 저장

#include <TensorFlowLite.h>
#include "time_measurements.h"

#include "main_functions.h"
#include "mnist_model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_profiler.h"

//extern float x_test[]; //테스트 이미지 5

unsigned long inference_time; //전체 추론 시간

const int kInputTensorSize = 1 * 784;
const int kNumClass = 10;

// Globals, used for compatibility with Arduino-style sketches.
namespace {
  const tflite::Model *model = nullptr;
  tflite::MicroInterpreter *interpreter = nullptr;
  TfLiteTensor *input = nullptr;
  constexpr int kTensorArenaSize = 100 * 1024; 
  alignas(16) uint8_t tensor_arena[kTensorArenaSize];
  // Profiler
  tflite::MicroProfiler micro_profiler;
} 


void profile_print(const char* func_name, unsigned long func_time, unsigned long func_num) {
  char buf[80];
  sprintf(buf, "%-37s : %-14lu | %-8lu | %-8lu", func_name, func_time, func_num, func_time*func_num);
  Serial.println(buf);
}

// The name of this function is important for Arduino compatibility.
void setup() {
  Serial.begin(9600);

  tflite::InitializeTarget();

  model = tflite::GetModel(g_person_detect_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf(
      "Model provided is schema version %d not equal "
      "to supported version %d.",
      model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  static tflite::MicroMutableOpResolver<10> micro_op_resolver;
  micro_op_resolver.AddShape();
  micro_op_resolver.AddStridedSlice();  //use
  micro_op_resolver.AddPack();          //use
  micro_op_resolver.AddMaxPool2D();
  micro_op_resolver.AddFullyConnected();
  //micro_op_resolver.AddAveragePool2D();   //no use
  micro_op_resolver.AddConv2D();
  //micro_op_resolver.AddDepthwiseConv2D(); //no use
  micro_op_resolver.AddReshape();
  //micro_op_resolver.AddSoftmax();         //no use

  static tflite::MicroInterpreter static_interpreter(
    model, micro_op_resolver, tensor_arena, kTensorArenaSize, nullptr,  &micro_profiler);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }
  input = interpreter->input(0);
}



// The name of this function is important for Arduino compatibility.
void loop() {

  float x_test[kInputTensorSize] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.32941177, 0.7254902, 0.62352943, 0.5921569, 0.23529412, 0.14117648, 0.0, 0.0, 0.0,
                                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.87058824, 0.99607843, 0.99607843,
                                     0.99607843, 0.99607843, 0.94509804, 0.7764706, 0.7764706, 0.7764706, 0.7764706, 0.7764706, 0.7764706, 0.7764706, 0.7764706, 0.6666667,
                                     0.20392157, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2627451, 0.44705883, 0.28235295, 0.44705883, 0.6392157, 0.8901961,
                                     0.99607843, 0.88235295, 0.99607843, 0.99607843, 0.99607843, 0.98039216, 0.8980392, 0.99607843, 0.99607843, 0.54901963, 0.0, 0.0, 0.0, 0.0,
                                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06666667, 0.25882354, 0.05490196, 0.2627451, 0.2627451, 0.2627451, 0.23137255,
                                     0.08235294, 0.9254902, 0.99607843, 0.41568628, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                     0.0, 0.0, 0.0, 0.0, 0.3254902, 0.99215686, 0.81960785, 0.07058824, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.08627451, 0.9137255, 1.0, 0.3254902, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5058824, 0.99607843, 0.93333334, 0.17254902, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.23137255, 0.9764706, 0.99607843, 0.24313726, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.52156866, 0.99607843, 0.73333335, 0.019607844, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03529412, 0.8039216, 0.972549, 0.22745098, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.49411765, 0.99607843, 0.7137255, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.29411766, 0.9843137, 0.9411765, 0.22352941, 0.0, 0.0, 0.0, 0.0,
                                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.07450981, 0.8666667, 0.99607843, 0.6509804, 0.0, 0.0, 0.0,
                                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.011764706, 0.79607844, 0.99607843, 0.85882354, 0.13725491,
                                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.14901961, 0.99607843, 0.99607843, 0.3019608,
                                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.12156863, 0.8784314, 0.99607843, 0.4509804,
                                     0.003921569, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.52156866, 0.99607843, 0.99607843,
                                     0.20392157, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.23921569, 0.9490196, 0.99607843,
                                     0.99607843, 0.20392157, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4745098, 0.99607843,
                                     0.99607843, 0.85882354, 0.15686275, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4745098,
                                     0.99607843, 0.8117647, 0.07058824, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };



// 전처리: 입력 데이터를 signed int8로 양자화
  int8_t quantized_input[kInputTensorSize];
  preprocess(x_test, quantized_input, kInputTensorSize, input->params.scale, input->params.zero_point);
  memcpy(input->data.int8, quantized_input, kInputTensorSize * sizeof(int8_t));

  uint32_t event_handle = micro_profiler.BeginEvent("Invoke");
  // Run the model on this input and make sure it succeeds.
  unsigned long start_time = micros();  // 코드 실행 시작 시간 기록
  if (kTfLiteOk != interpreter->Invoke()) {
    MicroPrintf("Invoke failed.");
  }
  unsigned long end_time = micros();  // 코드 실행 종료 시간 기록
  micro_profiler.EndEvent(event_handle);





  TfLiteTensor* output = interpreter->output(0);

  // int8 값을 그대로 출력
  int8_t* quantized_output = output->data.int8;
  Serial.println("Output probabilities (int8):");
  for (int i = 0; i < kNumClass; i++) {
    Serial.print("Class ");
    Serial.print(i);
    Serial.print(": ");
    Serial.println(quantized_output[i]);
  }

  int predicated_class = 0;
  int8_t max_score = -128;
  for (int i = 0; i < kNumClass; i++) {
    int8_t score = quantized_output[i];
    if (score > max_score) {
      predicated_class = i;
      max_score = score;
    }
  }


///디버깅

  Serial.println("\n***************************************************************************");
  Serial.print("predicated_class : ");
  Serial.println(predicated_class);

  inference_time = end_time - start_time;  // 추론 시간 계산
  Serial.print("total time : ");
  Serial.println(inference_time);

  Serial.println("\n============================================================================");
  Serial.println("function                              : (micros/call)  | call     | time    ");
  Serial.println("----------------------------------------------------------------------------");
  profile_print("InvokeSubgraph", InvokeSubgraph_time, InvokeSubgraph_num);
  profile_print("InvokeSubgraph_NumSubgraphOperators", InvokeSubgraph_NumSubgraphOperators_time, InvokeSubgraph_NumSubgraphOperators_num);
  profile_print("InvokeSubgraph_for", InvokeSubgraph_for_time, InvokeSubgraph_for_num);
  profile_print("InvokeSubgraph_declare_node", InvokeSubgraph_declare_node_time, InvokeSubgraph_declare_node_num);
  profile_print("InvokeSubgraph_declare", InvokeSubgraph_declare_registration_time, InvokeSubgraph_declare_registration_num);
  profile_print("InvokeSubgraph_error_priflier", InvokeSubgraph_error_priflier_time, InvokeSubgraph_error_priflier_num);
  
  unsigned long t1 = 0;
  for(int i=0;i<8;i++) t1 +=InvokeSubgraph_operation_time[i];
  profile_print("InvokeSubgraph_operation", t1, 1 );

  for(int i=0;i<8;i++){
    char func_name[50];
    snprintf(func_name, sizeof(func_name), "  >>  InvokeSubgraph_operation[%d]", i);
    profile_print(func_name, InvokeSubgraph_operation_time[i], 1);
  }
  
  profile_print("InvokeSubgraph_ResetTempAllocations", InvokeSubgraph_ResetTempAllocations_time, InvokeSubgraph_ResetTempAllocations_num);
  profile_print("InvokeSubgraph_kTfLiteError", InvokeSubgraph_kTfLiteError_time, InvokeSubgraph_kTfLiteError_num);

  Serial.println("============================================================================");
  micro_profiler.Log();
  Serial.println("****************************************************************************");
  

  //전역변수 디버깅
  /*
  for(int i=0;i<10;i++){
  char name[10];
  snprintf(name, sizeof(name), "t[%d]", i);
  Serial.print(name);
  Serial.print("  :  ");
  Serial.println(t[i]);
  }
  Serial.println("****************************************************************************");
  */

  reset_measurements();
  exit(0);
}