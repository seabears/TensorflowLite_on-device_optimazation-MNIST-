/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/micro_graph.h"

#include "third_party/flatbuffers/include/flatbuffers/flatbuffers.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/micro/flatbuffer_utils.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/schema/schema_generated.h"


#include "time_measurements.h"

namespace tflite {
namespace {

const char* OpNameFromRegistration(const TfLiteRegistration* registration) {
  if (registration->builtin_code == BuiltinOperator_CUSTOM) {
    return registration->custom_name;
  } else {
    return EnumNameBuiltinOperator(BuiltinOperator(registration->builtin_code));
  }
}

}  // namespace

MicroGraph::MicroGraph(TfLiteContext* context, const Model* model,
                       MicroAllocator* allocator,
                       MicroResourceVariables* resource_variables)
    : context_(context),
      model_(model),
      allocator_(allocator),
      current_subgraph_index_(0),
      resource_variables_(resource_variables) {
  if (model != nullptr) {
    subgraphs_ = model->subgraphs();
  }
}

MicroGraph::~MicroGraph() {}

TfLiteStatus MicroGraph::InitSubgraphs() {
  int previous_subgraph_idx = current_subgraph_index_;

  for (size_t subgraph_idx = 0; subgraph_idx < subgraphs_->size();
       subgraph_idx++) {
    current_subgraph_index_ = subgraph_idx;
    uint32_t operators_size = NumSubgraphOperators(model_, subgraph_idx);
    for (size_t i = 0; i < operators_size; ++i) {
      TfLiteNode* node =
          &(subgraph_allocations_[subgraph_idx].node_and_registrations[i].node);
      const TfLiteRegistration* registration =
          subgraph_allocations_[subgraph_idx]
              .node_and_registrations[i]
              .registration;
      size_t init_data_size;
      const char* init_data;
      if (registration->builtin_code == BuiltinOperator_CUSTOM) {
        init_data = reinterpret_cast<const char*>(node->custom_initial_data);
        init_data_size = node->custom_initial_data_size;
      } else {
        init_data = reinterpret_cast<const char*>(node->builtin_data);
        init_data_size = 0;
      }
      if (registration->init) {
        node->user_data =
            registration->init(context_, init_data, init_data_size);
      }
    }
  }
  current_subgraph_index_ = previous_subgraph_idx;

  return kTfLiteOk;
}

TfLiteStatus MicroGraph::PrepareSubgraphs() {
  int previous_subgraph_idx = current_subgraph_index_;

  for (size_t subgraph_idx = 0; subgraph_idx < subgraphs_->size();
       subgraph_idx++) {
    current_subgraph_index_ = subgraph_idx;
    uint32_t operators_size = NumSubgraphOperators(model_, subgraph_idx);
    for (size_t i = 0; i < operators_size; ++i) {
      TfLiteNode* node =
          &(subgraph_allocations_[subgraph_idx].node_and_registrations[i].node);
      const TfLiteRegistration* registration =
          subgraph_allocations_[subgraph_idx]
              .node_and_registrations[i]
              .registration;
      if (registration->prepare != nullptr) {
        TfLiteStatus prepare_status = registration->prepare(context_, node);
        if (prepare_status != kTfLiteOk) {
          MicroPrintf("Node %s (number %df) failed to prepare with status %d",
                      OpNameFromRegistration(registration), i, prepare_status);
          return kTfLiteError;
        }
      }
      allocator_->FinishPrepareNodeAllocations(/*node_id=*/i);
    }
  }
  current_subgraph_index_ = previous_subgraph_idx;

  return kTfLiteOk;
}

TfLiteStatus MicroGraph::FreeSubgraphs() {
  int previous_subgraph_idx = current_subgraph_index_;

  for (size_t subgraph_idx = 0; subgraph_idx < subgraphs_->size();
       subgraph_idx++) {
    current_subgraph_index_ = subgraph_idx;
    uint32_t operators_size = NumSubgraphOperators(model_, subgraph_idx);
    for (size_t i = 0; i < operators_size; ++i) {
      TfLiteNode* node =
          &(subgraph_allocations_[subgraph_idx].node_and_registrations[i].node);
      const TfLiteRegistration* registration =
          subgraph_allocations_[subgraph_idx]
              .node_and_registrations[i]
              .registration;
      // registration is allocated outside the interpreter, so double check to
      // make sure it's not nullptr;
      if (registration != nullptr && registration->free != nullptr) {
        registration->free(context_, node->user_data);
      }
    }
  }
  current_subgraph_index_ = previous_subgraph_idx;

  return kTfLiteOk;
}









TfLiteStatus MicroGraph::InvokeSubgraph(int subgraph_idx) { //subgraph_idx = 0
  int previous_subgraph_idx = current_subgraph_index_;  //임시 저장
  current_subgraph_index_ = subgraph_idx;

  if (static_cast<size_t>(subgraph_idx) >= subgraphs_->size()) { //서브그래프 인덱스가 유효한지 확인
    MicroPrintf("Accessing subgraph %d but only %d subgraphs found",
                subgraph_idx, subgraphs_->size());
    return kTfLiteError;
  }

  InvokeSubgraph_NumSubgraphOperators_num++;
  unsigned long start1 = micros();
  uint32_t operators_size = NumSubgraphOperators(model_, subgraph_idx); //subgraph_idx=0, 모델의 0번째 오버레이터 
  unsigned long end1 = micros();
  InvokeSubgraph_NumSubgraphOperators_time = end1 - start1;


  //Serial.println(operators_size);

  unsigned long start2 = micros();
  InvokeSubgraph_for_num++;
  for (size_t i = 0; i < operators_size; ++i) {           //operatiors_size = 8

    unsigned long start3 = micros();
    TfLiteNode* node =                                      //현재 오퍼레이션의 노드 갖고옴
        &(subgraph_allocations_[subgraph_idx].node_and_registrations[i].node);
    unsigned long end3 = micros();
    InvokeSubgraph_declare_node_time = (end3 - start3);
    InvokeSubgraph_declare_node_num++;


    unsigned long start4 = micros();
    const TfLiteRegistration* registration = subgraph_allocations_[subgraph_idx]    //현재 오퍼레이션의 등록 정보
                                                 .node_and_registrations[i]
                                                 .registration;
    unsigned long end4 = micros();
    InvokeSubgraph_declare_registration_time = end4 - start4;
    InvokeSubgraph_declare_registration_num++;

// This ifdef is needed (even though ScopedMicroProfiler itself is a no-op with
// -DTF_LITE_STRIP_ERROR_STRINGS) because the function OpNameFromRegistration is
// only defined for builds with the error strings.
    unsigned long start5 = micros(); 
#if !defined(TF_LITE_STRIP_ERROR_STRINGS)                                                   //아니면 현재 오퍼레이션으로 프로파일링 시작
    ScopedMicroProfiler scoped_profiler(
        OpNameFromRegistration(registration),
        reinterpret_cast<MicroProfilerInterface*>(context_->profiler));
#endif
    unsigned long end5 = micros(); 
    InvokeSubgraph_error_priflier_time = end5 - start5;
    InvokeSubgraph_error_priflier_num++;


       
    TFLITE_DCHECK(registration->invoke);    
    unsigned long start6 = micros();                                 
    TfLiteStatus invoke_status = registration->invoke(context_, node);    //오퍼레이션 실행
    unsigned long end6 = micros(); 
    InvokeSubgraph_operation_time[i] = end6-start6;
    InvokeSubgraph_operation_num++;
    
    unsigned long start7 = micros(); 
    allocator_->ResetTempAllocations();     //할당초기화
    unsigned long end7 = micros(); 
    InvokeSubgraph_ResetTempAllocations_time = end7 - start7;
    InvokeSubgraph_ResetTempAllocations_num++;

    unsigned long start8 = micros(); 
    if (invoke_status == kTfLiteError) {
      MicroPrintf("Node %s (number %d) failed to invoke with status %d",
                  OpNameFromRegistration(registration), i, invoke_status);
      return kTfLiteError;
    } else if (invoke_status != kTfLiteOk) {
      return invoke_status;
    }
    unsigned long end8 = micros(); 
    InvokeSubgraph_kTfLiteError_time = end8 - start8;
    InvokeSubgraph_kTfLiteError_num++;
  }
  unsigned long end2 = micros();  
  InvokeSubgraph_for_time = end2 - start2;

  current_subgraph_index_ = previous_subgraph_idx; 
  return kTfLiteOk;
}









TfLiteStatus MicroGraph::ResetVariableTensors() {
  for (size_t subgraph_idx = 0; subgraph_idx < subgraphs_->size();
       subgraph_idx++) {
    const SubGraph* subgraph = (*subgraphs_)[subgraph_idx];
    for (size_t i = 0; i < subgraph->tensors()->size(); ++i) {
      auto* tensor = subgraph->tensors()->Get(i);
      if (tensor->is_variable()) {
        size_t buffer_size;
        TF_LITE_ENSURE_STATUS(TfLiteEvalTensorByteLength(
            &subgraph_allocations_[subgraph_idx].tensors[i], &buffer_size));

        int value = 0;
        if (tensor->type() == tflite::TensorType_INT8) {
          value = tensor->quantization()->zero_point()->Get(0);
        }
        memset(subgraph_allocations_[subgraph_idx].tensors[i].data.raw, value,
               buffer_size);
      }
    }
  }
  if (resource_variables_ != nullptr) {
    resource_variables_->ResetAll();
  }

  return kTfLiteOk;
}

int MicroGraph::NumSubgraphs() { return model_->subgraphs()->size(); }

void MicroGraph::SetSubgraphAllocations(
    SubgraphAllocations* subgraph_allocations) {
  subgraph_allocations_ = subgraph_allocations;
}

size_t MicroGraph::NumSubgraphInputs(int subgraph_idx) {
  return model_->subgraphs()->Get(subgraph_idx)->inputs()->size();
}

TfLiteEvalTensor* MicroGraph::GetSubgraphInput(int subgraph_idx,
                                               int input_idx) {
  int tensor_idx =
      model_->subgraphs()->Get(subgraph_idx)->inputs()->Get(input_idx);
  return &subgraph_allocations_[subgraph_idx].tensors[tensor_idx];
}

size_t MicroGraph::NumSubgraphOutputs(int subgraph_idx) {
  return model_->subgraphs()->Get(subgraph_idx)->outputs()->size();
}

TfLiteEvalTensor* MicroGraph::GetSubgraphOutput(int subgraph_idx,
                                                int output_idx) {
  int tensor_idx =
      model_->subgraphs()->Get(subgraph_idx)->outputs()->Get(output_idx);
  return &subgraph_allocations_[subgraph_idx].tensors[tensor_idx];
}

}  // namespace tflite
