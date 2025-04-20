/*
 * SPDX-FileCopyrightText: Copyright 2020-2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Title:        arm_nn_vec_mat_mult_t_s8
 * Description:  s8 vector by matrix (transposed) multiplication
 *
 * $Date:        16 Aug 2022
 * $Revision:    V.4.0.2
 *
 * Target Processor:  Cortex-M
 *
 * -------------------------------------------------------------------- */

#include "third_party/cmsis_nn/Include/arm_nnsupportfunctions.h"

#include "time_measurements.h"
/**
 * @ingroup groupSupport
 */

/**
 * @defgroup supportFC Fully Connected
 *
 * Support functions for Fully Connected
 *
 */

/**
 * @addtogroup supportFC
 * @{
 */

/*
 * s8 vector(lhs) by matrix (transposed) multiplication
 *
 * Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_nn_vec_mat_mult_t_s8(const q7_t *lhs,
                                             const q7_t *rhs,
                                             const q31_t *bias,
                                             q7_t *dst,
                                             const int32_t lhs_offset,
                                             const int32_t rhs_offset,
                                             const int32_t dst_offset,
                                             const int32_t dst_multiplier,
                                             const int32_t dst_shift,
                                             const int32_t rhs_cols,
                                             const int32_t rhs_rows,
                                             const int32_t activation_min,
                                             const int32_t activation_max,
                                             const int32_t address_offset)
{
    (void)rhs_offset;
//#if defined(ARM_MATH_MVEI)
    
//#elif defined(ARM_MATH_DSP)
    //n[2]++;

    const int32_t row_loop_cnt = rhs_rows / 2;
    const int16_t lhs_offset_s16 = (int16_t)lhs_offset;
    const uint32_t lhs_offset_s16x2 = __PKHBT(lhs_offset_s16, lhs_offset_s16, 16);
    
    //s[2] = micros();    //  1947 / 2058
    /*
    for (int32_t i = 0; i < row_loop_cnt; i++)  //5번
    {   //n[2]++; //5
        //n[2] = row_loop_cnt;

        int32_t acc_0 = 0;
        int32_t acc_1 = 0;
        if (bias)
        {
            acc_0 = *bias++;
            acc_1 = *bias++;
        }

        const int32_t col_loop_cnt = rhs_cols / 4;

        const int8_t *lhs_vec = lhs;
        const int8_t *rhs_0 = rhs;
        const int8_t *rhs_1 = rhs + rhs_cols;
        rhs += 2 * rhs_cols;

        for (int j = col_loop_cnt; j != 0; j--) //507 * 5 = 2535번
        {   //n[3]++; //2535
            //n[3] = col_loop_cnt;

            int32_t vec_0 = arm_nn_read_q7x4_ia(&lhs_vec);
            int32_t vec_1 = __SXTAB16_RORn(lhs_offset_s16x2, (uint32_t)vec_0, 8);

            vec_0 = __SXTAB16(lhs_offset_s16x2, vec_0);

            int32_t ker_0 = arm_nn_read_q7x4_ia(&rhs_0);
            int32_t ker_1 = __SXTB16_RORn((uint32_t)ker_0, 8);
            ker_0 = __SXTB16(ker_0);

            acc_0 = __SMLAD(ker_1, vec_1, acc_0);
            acc_0 = __SMLAD(ker_0, vec_0, acc_0);

            ker_0 = arm_nn_read_q7x4_ia(&rhs_1);
            ker_1 = __SXTB16_RORn((uint32_t)ker_0, 8);
            ker_0 = __SXTB16(ker_0);

            acc_1 = __SMLAD(ker_1, vec_1, acc_1);
            acc_1 = __SMLAD(ker_0, vec_0, acc_1);
        }


        acc_0 = arm_nn_requantize(acc_0, dst_multiplier, dst_shift);
        acc_1 = arm_nn_requantize(acc_1, dst_multiplier, dst_shift);

        // Add offset
        acc_0 += dst_offset;
        acc_1 += dst_offset;
        // Clamp the result
        acc_0 = MAX(acc_0, activation_min);
        acc_0 = MIN(acc_0, activation_max);
        acc_1 = MAX(acc_1, activation_min);
        acc_1 = MIN(acc_1, activation_max);
        *dst = (int8_t)acc_0;
        *(dst + address_offset) = (int8_t)acc_1;
        dst += 2 * address_offset;
    }*/




for (int32_t i = 0; i < row_loop_cnt; i++)  // 5번 반복
{
    int32_t acc_0 = 0;
    int32_t acc_1 = 0;
    if (bias)
    {
        acc_0 = *bias++;
        acc_1 = *bias++;
    }

    const int32_t col_loop_cnt = rhs_cols / 32;  // 동시처리 4 , 루프언롤링 8

    const int8_t *lhs_vec = lhs;
    const int8_t *rhs_0 = rhs;
    const int8_t *rhs_1 = rhs + rhs_cols;
    rhs += 2 * rhs_cols;

    for (int j = col_loop_cnt; j != 0; j--)  // 루프 언롤링 적용
    {
        int32_t vec_0, vec_1, ker_0, ker_1;

        // 첫 번째 그룹
        vec_0 = arm_nn_read_q7x4_ia(&lhs_vec);
        vec_1 = __SXTAB16_RORn(lhs_offset_s16x2, (uint32_t)vec_0, 8);
        vec_0 = __SXTAB16(lhs_offset_s16x2, vec_0);
        ker_0 = arm_nn_read_q7x4_ia(&rhs_0);
        ker_1 = __SXTB16_RORn((uint32_t)ker_0, 8);
        ker_0 = __SXTB16(ker_0);
        acc_0 = __SMLAD(ker_1, vec_1, acc_0);
        acc_0 = __SMLAD(ker_0, vec_0, acc_0);
        ker_0 = arm_nn_read_q7x4_ia(&rhs_1);
        ker_1 = __SXTB16_RORn((uint32_t)ker_0, 8);
        ker_0 = __SXTB16(ker_0);
        acc_1 = __SMLAD(ker_1, vec_1, acc_1);
        acc_1 = __SMLAD(ker_0, vec_0, acc_1);

        // 두 번째 그룹
        vec_0 = arm_nn_read_q7x4_ia(&lhs_vec);
        vec_1 = __SXTAB16_RORn(lhs_offset_s16x2, (uint32_t)vec_0, 8);
        vec_0 = __SXTAB16(lhs_offset_s16x2, vec_0);
        ker_0 = arm_nn_read_q7x4_ia(&rhs_0);
        ker_1 = __SXTB16_RORn((uint32_t)ker_0, 8);
        ker_0 = __SXTB16(ker_0);
        acc_0 = __SMLAD(ker_1, vec_1, acc_0);
        acc_0 = __SMLAD(ker_0, vec_0, acc_0);
        ker_0 = arm_nn_read_q7x4_ia(&rhs_1);
        ker_1 = __SXTB16_RORn((uint32_t)ker_0, 8);
        ker_0 = __SXTB16(ker_0);
        acc_1 = __SMLAD(ker_1, vec_1, acc_1);
        acc_1 = __SMLAD(ker_0, vec_0, acc_1);

        // 세 번째 그룹
        vec_0 = arm_nn_read_q7x4_ia(&lhs_vec);
        vec_1 = __SXTAB16_RORn(lhs_offset_s16x2, (uint32_t)vec_0, 8);
        vec_0 = __SXTAB16(lhs_offset_s16x2, vec_0);
        ker_0 = arm_nn_read_q7x4_ia(&rhs_0);
        ker_1 = __SXTB16_RORn((uint32_t)ker_0, 8);
        ker_0 = __SXTB16(ker_0);
        acc_0 = __SMLAD(ker_1, vec_1, acc_0);
        acc_0 = __SMLAD(ker_0, vec_0, acc_0);
        ker_0 = arm_nn_read_q7x4_ia(&rhs_1);
        ker_1 = __SXTB16_RORn((uint32_t)ker_0, 8);
        ker_0 = __SXTB16(ker_0);
        acc_1 = __SMLAD(ker_1, vec_1, acc_1);
        acc_1 = __SMLAD(ker_0, vec_0, acc_1);

        // 네 번째 그룹
        vec_0 = arm_nn_read_q7x4_ia(&lhs_vec);
        vec_1 = __SXTAB16_RORn(lhs_offset_s16x2, (uint32_t)vec_0, 8);
        vec_0 = __SXTAB16(lhs_offset_s16x2, vec_0);
        ker_0 = arm_nn_read_q7x4_ia(&rhs_0);
        ker_1 = __SXTB16_RORn((uint32_t)ker_0, 8);
        ker_0 = __SXTB16(ker_0);
        acc_0 = __SMLAD(ker_1, vec_1, acc_0);
        acc_0 = __SMLAD(ker_0, vec_0, acc_0);
        ker_0 = arm_nn_read_q7x4_ia(&rhs_1);
        ker_1 = __SXTB16_RORn((uint32_t)ker_0, 8);
        ker_0 = __SXTB16(ker_0);
        acc_1 = __SMLAD(ker_1, vec_1, acc_1);
        acc_1 = __SMLAD(ker_0, vec_0, acc_1);

        // 다섯 번째 그룹
        vec_0 = arm_nn_read_q7x4_ia(&lhs_vec);
        vec_1 = __SXTAB16_RORn(lhs_offset_s16x2, (uint32_t)vec_0, 8);
        vec_0 = __SXTAB16(lhs_offset_s16x2, vec_0);
        ker_0 = arm_nn_read_q7x4_ia(&rhs_0);
        ker_1 = __SXTB16_RORn((uint32_t)ker_0, 8);
        ker_0 = __SXTB16(ker_0);
        acc_0 = __SMLAD(ker_1, vec_1, acc_0);
        acc_0 = __SMLAD(ker_0, vec_0, acc_0);
        ker_0 = arm_nn_read_q7x4_ia(&rhs_1);
        ker_1 = __SXTB16_RORn((uint32_t)ker_0, 8);
        ker_0 = __SXTB16(ker_0);
        acc_1 = __SMLAD(ker_1, vec_1, acc_1);
        acc_1 = __SMLAD(ker_0, vec_0, acc_1);

        // 여섯 번째 그룹
        vec_0 = arm_nn_read_q7x4_ia(&lhs_vec);
        vec_1 = __SXTAB16_RORn(lhs_offset_s16x2, (uint32_t)vec_0, 8);
        vec_0 = __SXTAB16(lhs_offset_s16x2, vec_0);
        ker_0 = arm_nn_read_q7x4_ia(&rhs_0);
        ker_1 = __SXTB16_RORn((uint32_t)ker_0, 8);
        ker_0 = __SXTB16(ker_0);
        acc_0 = __SMLAD(ker_1, vec_1, acc_0);
        acc_0 = __SMLAD(ker_0, vec_0, acc_0);
        ker_0 = arm_nn_read_q7x4_ia(&rhs_1);
        ker_1 = __SXTB16_RORn((uint32_t)ker_0, 8);
        ker_0 = __SXTB16(ker_0);
        acc_1 = __SMLAD(ker_1, vec_1, acc_1);
        acc_1 = __SMLAD(ker_0, vec_0, acc_1);

        // 일곱 번째 그룹
        vec_0 = arm_nn_read_q7x4_ia(&lhs_vec);
        vec_1 = __SXTAB16_RORn(lhs_offset_s16x2, (uint32_t)vec_0, 8);
        vec_0 = __SXTAB16(lhs_offset_s16x2, vec_0);
        ker_0 = arm_nn_read_q7x4_ia(&rhs_0);
        ker_1 = __SXTB16_RORn((uint32_t)ker_0, 8);
        ker_0 = __SXTB16(ker_0);
        acc_0 = __SMLAD(ker_1, vec_1, acc_0);
        acc_0 = __SMLAD(ker_0, vec_0, acc_0);
        ker_0 = arm_nn_read_q7x4_ia(&rhs_1);
        ker_1 = __SXTB16_RORn((uint32_t)ker_0, 8);
        ker_0 = __SXTB16(ker_0);
        acc_1 = __SMLAD(ker_1, vec_1, acc_1);
        acc_1 = __SMLAD(ker_0, vec_0, acc_1);

        // 여덟 번째 그룹
        vec_0 = arm_nn_read_q7x4_ia(&lhs_vec);
        vec_1 = __SXTAB16_RORn(lhs_offset_s16x2, (uint32_t)vec_0, 8);
        vec_0 = __SXTAB16(lhs_offset_s16x2, vec_0);
        ker_0 = arm_nn_read_q7x4_ia(&rhs_0);
        ker_1 = __SXTB16_RORn((uint32_t)ker_0, 8);
        ker_0 = __SXTB16(ker_0);
        acc_0 = __SMLAD(ker_1, vec_1, acc_0);
        acc_0 = __SMLAD(ker_0, vec_0, acc_0);
        ker_0 = arm_nn_read_q7x4_ia(&rhs_1);
        ker_1 = __SXTB16_RORn((uint32_t)ker_0, 8);
        ker_0 = __SXTB16(ker_0);
        acc_1 = __SMLAD(ker_1, vec_1, acc_1);
        acc_1 = __SMLAD(ker_0, vec_0, acc_1);

    }

    // 나머지 데이터 처리
    for (int k = col_loop_cnt * 32; k < rhs_cols; k++)
    {
        const int32_t lhs_temp = (*lhs_vec + lhs_offset);
        lhs_vec++;
        acc_0 += lhs_temp * (*rhs_0);
        rhs_0++;
        acc_1 += lhs_temp * (*rhs_1);
        rhs_1++;
    }

    acc_0 = arm_nn_requantize(acc_0, dst_multiplier, dst_shift);
    acc_1 = arm_nn_requantize(acc_1, dst_multiplier, dst_shift);

    // Add offset
    acc_0 += dst_offset;
    acc_1 += dst_offset;
    // Clamp the result
    acc_0 = MAX(acc_0, activation_min);
    acc_0 = MIN(acc_0, activation_max);
    acc_1 = MAX(acc_1, activation_min);
    acc_1 = MIN(acc_1, activation_max);
    *dst = (int8_t)acc_0;
    *(dst + address_offset) = (int8_t)acc_1;
    dst += 2 * address_offset;
}








   ////
    //t[2] = micros() - s[2];



//#else

//#endif
    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Doxygen group
 */
