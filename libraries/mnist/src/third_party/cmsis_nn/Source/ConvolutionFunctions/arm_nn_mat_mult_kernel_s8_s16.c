/*
 * Copyright (C) 2010-2021 Arm Limited or its affiliates. All rights reserved.
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
 * Title:        arm_nn_mat_mult_kernel_s8_s16.c
 * Description:  Matrix-multiplication function for convolution
 *
 * $Date:        14. December 2021
 * $Revision:    V.1.1.0
 *
 * Target Processor:  Cortex-M cores
 * -------------------------------------------------------------------- */


#define REQUANTIZE_AND_CLAMP_AND_STORE(ch_out, out_mult_ptr, out_shift_ptr, out_offset, activation_min, activation_max, out_ptr) \
        ch_out = arm_nn_requantize(ch_out, *out_mult_ptr, *out_shift_ptr); \
        ch_out += out_offset; \
        ch_out = MAX(ch_out, activation_min); \
        ch_out = MIN(ch_out, activation_max); \
        *(out_ptr)++ = (q7_t)ch_out;
        //test[n[1]++] = ch_out;
     
        //if((n[4]++) == 0)  n[5] = (q7_t)ch_out;
        //n[4]++; if(ch_out == -128) n[2]++; //4294967168
        //8112 = 13*26    * 6     *4
        //6195 // 8112 동안 ch_out = -128


#include "third_party/cmsis_nn/Include/arm_nnfunctions.h"
#include "third_party/cmsis_nn/Include/arm_nnsupportfunctions.h"

#include "time_measurements.h"
/*
 * Matrix-multiplication function for convolution with per-channel requantization.
 *
 * Refer header file for details.
 *
 */



q7_t *arm_nn_mat_mult_kernel_s8_s16(const q7_t *input_a,
                                    const q15_t *input_b,
                                    const uint16_t output_ch,   //12 개의 출력 채널(필터 수)
                                    const int32_t *out_shift,
                                    const int32_t *out_mult,
                                    const int32_t out_offset,
                                    const int16_t activation_min,   //4294967168 = 0xFFFFFFF0 : -128
                                    const int16_t activation_max,   //127
                                    const uint16_t num_col_a,       //9
                                    const int32_t *const output_bias,
                                    q7_t *out_0)
{
#if !defined(ARM_MATH_MVEI)     //338번 호출 
    
    /* set up the second output pointers */
    q7_t *out_1 = out_0 + output_ch;
    const int32_t *bias = output_bias;

    uint16_t row_count = output_ch / 2;
    const q7_t *ip_a0 = input_a;
    /* this loop over rows in A */
    
    //row_count : 6
    while (row_count)   //2028번 반복
    {//s[5] = micros();   //19056/89016
                 
        /* setup pointers for B */
        const q15_t *ip_b0 = input_b;
        const q15_t *ip_b1 = ip_b0 + num_col_a;

        /* align the second pointer for A */
        const q7_t *ip_a1 = ip_a0 + num_col_a;

        q31_t ch_0_out_0 = 0;
        q31_t ch_0_out_1 = 0;
        q31_t ch_1_out_0 = 0;
        q31_t ch_1_out_1 = 0;
        /* Init accumulator with bias for channel N and N + 1 */
        if (bias)       //2028번
        {   
            ch_0_out_0 = *bias;
            ch_0_out_1 = *bias++;
            ch_1_out_0 = *bias;
            ch_1_out_1 = *bias++;
        }


        uint16_t col_count = num_col_a / 8; // num_col : 9 // col_count : 1
        q31_t a01, a02, a11, a12;
        q31_t a03, a04, a13, a14;
        q31_t b0, b1;

        // 첫 번째 4개 데이터 처리
        b0 = arm_nn_read_q15x2_ia(&ip_b0);
        b1 = arm_nn_read_q15x2_ia(&ip_b1);

        ip_a0 = read_and_pad(ip_a0, &a01, &a02);
        ip_a1 = read_and_pad(ip_a1, &a11, &a12);

        ch_0_out_0 = __SMLAD(a01, b0, ch_0_out_0);  //SMLAD : 16bit product //and add
        ch_0_out_1 = __SMLAD(a01, b1, ch_0_out_1);
        ch_1_out_0 = __SMLAD(a11, b0, ch_1_out_0);
        ch_1_out_1 = __SMLAD(a11, b1, ch_1_out_1);

        b0 = arm_nn_read_q15x2_ia(&ip_b0);
        b1 = arm_nn_read_q15x2_ia(&ip_b1);

        ch_0_out_0 = __SMLAD(a02, b0, ch_0_out_0);
        ch_0_out_1 = __SMLAD(a02, b1, ch_0_out_1);
        ch_1_out_0 = __SMLAD(a12, b0, ch_1_out_0);
        ch_1_out_1 = __SMLAD(a12, b1, ch_1_out_1);

        // 두 번째 4개 데이터 처리
        ip_a0 = read_and_pad(ip_a0, &a03, &a04);
        ip_a1 = read_and_pad(ip_a1, &a13, &a14);

        b0 = arm_nn_read_q15x2_ia(&ip_b0);
        b1 = arm_nn_read_q15x2_ia(&ip_b1);

        ch_0_out_0 = __SMLAD(a03, b0, ch_0_out_0);
        ch_0_out_1 = __SMLAD(a03, b1, ch_0_out_1);
        ch_1_out_0 = __SMLAD(a13, b0, ch_1_out_0);
        ch_1_out_1 = __SMLAD(a13, b1, ch_1_out_1);

        b0 = arm_nn_read_q15x2_ia(&ip_b0);
        b1 = arm_nn_read_q15x2_ia(&ip_b1);

        ch_0_out_0 = __SMLAD(a04, b0, ch_0_out_0);
        ch_0_out_1 = __SMLAD(a04, b1, ch_0_out_1);
        ch_1_out_0 = __SMLAD(a14, b0, ch_1_out_0);
        ch_1_out_1 = __SMLAD(a14, b1, ch_1_out_1);


        while (col_count)
        {                                //2028번
            q7_t a0 = *ip_a0++;
            q15_t b0 = *ip_b0++;
            q7_t a1 = *ip_a1++;
            q15_t b1 = *ip_b1++;

            ch_0_out_0 += a0 * b0;
            ch_0_out_1 += a0 * b1;
            ch_1_out_0 += a1 * b0;
            ch_1_out_1 += a1 * b1;
            col_count--;
        } // while over col_count 


REQUANTIZE_AND_CLAMP_AND_STORE(ch_0_out_0, out_mult, out_shift, out_offset, activation_min, activation_max, out_0);
REQUANTIZE_AND_CLAMP_AND_STORE(ch_0_out_1, out_mult, out_shift, out_offset, activation_min, activation_max, out_1);
out_mult++;
out_shift++;

REQUANTIZE_AND_CLAMP_AND_STORE(ch_1_out_0, out_mult, out_shift, out_offset, activation_min, activation_max, out_0);
REQUANTIZE_AND_CLAMP_AND_STORE(ch_1_out_1, out_mult, out_shift, out_offset, activation_min, activation_max, out_1);
out_mult++;
out_shift++;

        // skip row 
        ip_a0 += num_col_a;
        row_count--;
    }

    out_0 += output_ch;
    /* return the new output pointer with offset */
    return out_0;
#else
    (void)input_a;
    (void)input_b;
    (void)output_ch;
    (void)out_shift;
    (void)out_mult;
    (void)out_offset;
    (void)activation_min;
    (void)activation_max;
    (void)num_col_a;
    (void)output_bias;
    (void)out_0;
    /* To be completed */
    return NULL;
#endif


}
