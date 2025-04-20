/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstring>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/micro_utils.h"

#include "time_measurements.h"

namespace tflite {
namespace ops {
namespace micro {
namespace reshape {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

TfLiteStatus ReshapeOutput(TfLiteContext* context, TfLiteNode* node) {  //153 micros // 2번 호출
  //s[2] = micros();
  //s[4] = micros();      //71/224
  MicroContext* micro_context = GetMicroContext(context);
  
  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);
  // Tensorflow's Reshape allows one of the shape components to have the
  // special -1 value, meaning it will be calculated automatically based on the
  // input. Here we calculate what that dimension should be so that the number
  // of output elements in the same as the number of input elements.
  int num_input_elements = NumElements(input);
  TfLiteIntArray* output_shape = output->dims;

  if (NumInputs(node) == 1 &&  // Legacy scalar supported with params.
      output_shape->size == 1 && output_shape->data[0] == 0) {
    // Legacy tflite models use a shape parameter of [0] to indicate scalars,
    // so adjust accordingly. TODO(b/111614235): Allow zero-sized buffers during
    // toco conversion.
    output_shape->size = 0;
  }
  //t[4] = micros() - s[4];

  int num_output_elements = 1;
  int stretch_dim = -1;

  //s[4] = micros();
  for (int i = 0; i < output_shape->size; ++i) {  // 18/188 //총 6번 반복
    //n[5]++;
    int value = output_shape->data[i];
    if (value == -1) {
      TF_LITE_ENSURE_EQ(context, stretch_dim, -1);
      stretch_dim = i;
    } else {
      num_output_elements *= value;
    }
  }
  //t[4] += micros() - s[4];

  if (stretch_dim != -1) {    //안옴
    //n[4]++;
    TfLiteEvalTensor* output_eval =
        tflite::micro::GetEvalOutput(context, node, kOutputTensor);
    TF_LITE_ENSURE_STATUS(tflite::micro::CreateWritableTensorDimsWithCopy(
        context, output, output_eval));
    output_shape = output->dims;  // output tensor dims were moved
    output_shape->data[stretch_dim] = num_input_elements / num_output_elements;
    num_output_elements *= output_shape->data[stretch_dim];
  }
  //s[3] = micros();    //25 / 224
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  TF_LITE_ENSURE_EQ(context, num_input_elements, num_output_elements);

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(output);
  //t[3] += micros() - s[3];
  //t[2] += micros() - s[2];
  //n[2]++;
  return kTfLiteOk;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {    //96/153  //2번 호출
  //s[4] = micros();
  TF_LITE_ENSURE(context, NumInputs(node) == 1 || NumInputs(node) == 2);  //18
  //t[3] += micros() - s[4];
  //s[5] = micros();
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);    //17
  //t[5] += micros() - s[5];
  //s[6] = micros();
  TF_LITE_ENSURE_EQ(context, ReshapeOutput(context, node), kTfLiteOk);  //188
  //t[6] +=micros() - s[6];
  //n[4]++;
  //t[4] = micros() - s[4];
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) { //31 micros //2번 호출
  //s[3] = micros();
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  // TODO(b/162522304): storing input bytes in OpData increases some models
  // significantly, possibly due to alignment issues.
  size_t input_bytes;
  TF_LITE_ENSURE_STATUS(TfLiteTypeSizeOf(input->type, &input_bytes));
  input_bytes *= ElementCount(*input->dims);

  // Do nothing for in-place reshape.
  if (input->data.raw != output->data.raw) {
    // Otherwise perform reshape with copy.
    memcpy(output->data.raw, input->data.raw, input_bytes);
  }
  //t[3] = micros() - s[3];
  //n[3]++;
  return kTfLiteOk;
}

}  // namespace reshape

TfLiteRegistration Register_RESHAPE() {
  return tflite::micro::RegisterOp(nullptr, reshape::Prepare, reshape::Eval);
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
