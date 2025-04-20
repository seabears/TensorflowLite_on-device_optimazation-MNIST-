/*
 * Copyright (C) 2010-2022 Arm Limited or its affiliates.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_convolve_s8.c
 * Description:  s8 version of convolution using symmetric quantization.
 *
 * $Date:        19 April 2022
 * $Revision:    V.3.0.0
 *
 * Target Processor:  Cortex-M cores
 *
 * -------------------------------------------------------------------- */

#include "third_party/cmsis_nn/Include/arm_nnfunctions.h"
#include "third_party/cmsis_nn/Include/arm_nnsupportfunctions.h"

#include <stdint.h>
#include "time_measurements.h"
/**
 *  @ingroup Public
 */

/**
 * @addtogroup NNConv
 * @{
 */

/*
 * Basic s8 convolution function.
 *
 * Refer header file for details. Optimal use case for the DSP/MVE implementation is when input and output channels
 * are multiples of 4 or atleast greater than 4.
 *
 */

arm_cmsis_nn_status arm_convolve_3x3_s8(const cmsis_nn_context *ctx,
                                        const cmsis_nn_conv_params *conv_params,
                                        const cmsis_nn_per_channel_quant_params *quant_params,
                                        const cmsis_nn_dims *input_dims,
                                        const q7_t *input_data,
                                        const cmsis_nn_dims *filter_dims,
                                        const q7_t *filter_data,
                                        const cmsis_nn_dims *bias_dims,
                                        const int32_t *bias_data,
                                        const cmsis_nn_dims *output_dims,
                                        q7_t *output_data)
{
    (void)bias_dims;

    if (ctx->buf == NULL && arm_convolve_3x3_s8_get_buffer_size(input_dims, filter_dims) > 0)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }
    q15_t *buffer_a = (q15_t *)ctx->buf;
    // 입력데이터
    const int32_t input_batches = input_dims->n; // 1
    const uint16_t input_x = input_dims->w;      // 28
    const uint16_t input_y = input_dims->h;      // 28
    const uint16_t input_ch = input_dims->c;     // 1
    // 필터
    const uint16_t kernel_x = filter_dims->w; // 3
    const uint16_t kernel_y = filter_dims->h; // 3
    // 출력
    const uint16_t output_x = output_dims->w;  // 26
    const uint16_t output_y = output_dims->h;  // 26
    const uint16_t output_ch = output_dims->c; // 12?
    // 패딩
    const uint16_t pad_x = conv_params->padding.w; // 0
    const uint16_t pad_y = conv_params->padding.h; // 0
    // 스트라이드 : 필터의 이동 간격
    const uint16_t stride_x = conv_params->stride.w; // 1
    const uint16_t stride_y = conv_params->stride.h; // 1

    const int32_t input_offset = conv_params->input_offset; // 128
    const int32_t out_offset = conv_params->output_offset;
    const int32_t out_activation_min = conv_params->activation.min;
    const int32_t out_activation_max = conv_params->activation.max;
    int32_t *output_mult = quant_params->multiplier;
    int32_t *output_shift = quant_params->shift;


    // dilation : 필터(커널)이 입력 데이터를 훑을 때의 간격 조절에 사용
    const uint16_t dilation_x = conv_params->dilation.w; // 1
    const uint16_t dilation_y = conv_params->dilation.h; // 1

    int32_t i_out_y, i_out_x, i_ker_y, i_ker_x;

    // Generate two columns from the input tensor a GEMM computation 
    q15_t *two_column_buf = buffer_a;
    q7_t *out = output_data;

    // This part implements the im2col function
    for (i_out_y = 0; i_out_y < output_y; i_out_y++) // 26
    {
        for (i_out_x = 0; i_out_x < output_x; i_out_x++) // 26
        {
            // 현재 필터의 시작위치
            const int32_t base_idx_y = stride_y * i_out_y - pad_y;
            const int32_t base_idx_x = stride_x * i_out_x - pad_x;

            // 3*3 필터 크기만큼 반복    //
            const int32_t k_y1 = base_idx_y;
            const int32_t k_y2 = k_y1 + dilation_y;
            const int32_t k_y3 = k_y2 + dilation_y;

            const int32_t k_x1 = base_idx_x;
            const int32_t k_x2 = k_x1 + dilation_x;
            const int32_t k_x3 = k_x2 + dilation_x;

            const q7_t *src_ptrs[9] = {
                input_data + (k_y1 * input_x + k_x1),
                input_data + (k_y1 * input_x + k_x2),
                input_data + (k_y1 * input_x + k_x3),
                input_data + (k_y2 * input_x + k_x1),
                input_data + (k_y2 * input_x + k_x2),
                input_data + (k_y2 * input_x + k_x3),
                input_data + (k_y3 * input_x + k_x1),
                input_data + (k_y3 * input_x + k_x2),
                input_data + (k_y3 * input_x + k_x3)
                };

            q15_t *dst_ptr = two_column_buf;

            
            q7_t src1 = (*src_ptrs[0]);
            q7_t src2 = (*src_ptrs[1]);
            q7_t src3 = (*src_ptrs[2]);
            q7_t src4 = (*src_ptrs[3]);
            q7_t src5 = (*src_ptrs[4]);
            q7_t src6 = (*src_ptrs[5]);
            q7_t src7 = (*src_ptrs[6]);
            q7_t src8 = (*src_ptrs[7]);
            

            //simd사용했으나 조금 더 느림
            /*
            uint32_t mask1 = 0xFF;
            uint32_t mask2 = 0xFFFF;
            uint32_t summand = 128 + (128 << 16);

            uint32_t result1 = 0;
            uint32_t result2 = 0;
            uint32_t result3 = 0;
            uint32_t result4 = 0;    

            uint32_t val1 = src1 + (src2 << 16);
            uint32_t val2 = src3 + (src4 << 16);
            uint32_t val3 = src5 + (src6 << 16);
            uint32_t val4 = src7 + (src8 << 16);

            //UQADD16 -> SADD16
            result1 = __SADD16(val1, summand);
            result2 = __SADD16(val2, summand);
            result3 = __SADD16(val3, summand);
            result4 = __SADD16(val4, summand);

            *(dst_ptr++) = result1 & mask2;
            *(dst_ptr++) = (result1 >> 16) & mask2;
            *(dst_ptr++) = result2 & mask2;
            *(dst_ptr++) = (result2 >> 16) & mask2;
            *(dst_ptr++) = result3 & mask2;
            *(dst_ptr++) = (result3 >> 16) & mask2;
            *(dst_ptr++) = result4 & mask2;
            *(dst_ptr++) = (result4 >> 16) & mask2;            
            */
            
            *(dst_ptr++) = src1+128;
            *(dst_ptr++) = src2+128;
            *(dst_ptr++) = src3+128;
            *(dst_ptr++) = src4+128;
            *(dst_ptr++) = src5+128;
            *(dst_ptr++) = src6+128;
            *(dst_ptr++) = src7+128;
            *(dst_ptr++) = src8+128;
            

            // 남은 하나의 데이터 처리
            *dst_ptr++ = (q15_t)(*src_ptrs[8]++) + input_offset;

            two_column_buf += 9;
            
            //test[n[2]++] = *(input_data + n[2]);

            
            uint16_t num_col_a = input_ch * kernel_y * kernel_x; // 9
            // 2열 버퍼 채워지면 함수 호출
            //  Computation is filed for every 2 columns
            if (two_column_buf == buffer_a + 2 * num_col_a) // 18추가될 떄
            {

                //검은 부분 convolution 연산 최소화 //정확도 낮아짐
                /*
                bool all_zero = true;
                q15_t *tmp_ptr = buffer_a; //two_column_buf - 18
                // 첫 18개의 값이 모두 0인지 확인
                for (int i = 0; i < 18; i++) {  //4438
                   // n[3]++;
                    if ((*(tmp_ptr++)) != 0) {    //100
                        //n[2]++;
                        all_zero = false;
                        break;
                    }
                }

                // 모두 0일 경우 out 배열의 첫 24개의 값을 out_activation_min로 설정
                if (all_zero) {
                    for (int i = 1; i < 25; i++) {
                        *(out++) = out_activation_min;
                    }
                }

            else{*/
                                //338번 반복 : 338 = 13 * 26
                out = arm_nn_mat_mult_kernel_s8_s16(filter_data,
                                                    buffer_a,
                                                    output_ch,
                                                    output_shift,
                                                    output_mult,
                                                    out_offset,
                                                    out_activation_min,
                                                    out_activation_max,
                                                    num_col_a,
                                                    bias_data,
                                                    out);


            //}
                //  counter reset
                two_column_buf = buffer_a;
            }
        }
    }


    /* Advance to the next batch */
    input_data += (input_x * input_y * input_ch);
    output_data += (output_x * output_y * output_ch);

    /* Return to application */
    return ARM_CMSIS_NN_SUCCESS;
}

int32_t arm_convolve_3x3_s8_get_buffer_size(const cmsis_nn_dims *input_dims, const cmsis_nn_dims *filter_dims)
{
#if defined(ARM_MATH_MVEI)
    int32_t col_length = input_dims->c * filter_dims->w * filter_dims->h;
    // Get number of complete int16 lanes(multiple of 8) for given col_length. This is dependent on
    // implementation of  arm_nn_mat_mult_s8
    col_length = (col_length + 7) / 8;
    // 4 -> number of im2col buffers, 8 -> 8 elements per Q register
    return 4 * col_length * 8 * (int32_t)sizeof(int8_t);
#else
    return (2 * input_dims->c * filter_dims->w * filter_dims->h) * (int32_t)sizeof(int16_t);
#endif
}

/**
 * @} end of NNConv group
 */
