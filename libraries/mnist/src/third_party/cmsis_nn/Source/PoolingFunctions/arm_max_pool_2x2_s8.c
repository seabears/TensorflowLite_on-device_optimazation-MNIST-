#include "third_party/cmsis_nn/Include/arm_nnfunctions.h"
#include "third_party/cmsis_nn/Include/arm_nnsupportfunctions.h"

static void compare_and_replace_if_larger_q7_2x2(const q7_t *src1, const q7_t *src2, const q7_t *src3, const q7_t *src4, q7_t *dst, int32_t length)
{
    for (int i = 0; i < length; i++)
    {
        q7_t max_val = MAX(MAX(src1[i], src2[i]), MAX(src3[i], src4[i]));
        dst[i] = max_val;
    }
}

static void clamp_output_2x2(q7_t *source, int32_t length, const int32_t act_min, const int32_t act_max)
{
    union arm_nnword in;
    int32_t cnt = length >> 2;

    while (cnt > 0l)
    {
        in.word = arm_nn_read_q7x4(source);

        in.bytes[0] = MAX(in.bytes[0], act_min);
        in.bytes[0] = MIN(in.bytes[0], act_max);
        in.bytes[1] = MAX(in.bytes[1], act_min);
        in.bytes[1] = MIN(in.bytes[1], act_max);
        in.bytes[2] = MAX(in.bytes[2], act_min);
        in.bytes[2] = MIN(in.bytes[2], act_max);
        in.bytes[3] = MAX(in.bytes[3], act_min);
        in.bytes[3] = MIN(in.bytes[3], act_max);

        arm_nn_write_q7x4_ia(&source, in.word);
        cnt--;
    }

    cnt = length & 0x3;
    while (cnt > 0l)
    {
        int32_t comp = *source;
        comp = MAX(comp, act_min);
        comp = MIN(comp, act_max);
        *source++ = (int8_t)comp;
        cnt--;
    }
}

/**
 *  @ingroup Public
 */

/**
 * @addtogroup Pooling
 * @{
 */

/*
 * Optimized s8 max pooling function
 *
 * Refer to header file for details.
 *
 */

arm_cmsis_nn_status arm_max_pool_2x2_s8(const cmsis_nn_context *ctx,
                                        const cmsis_nn_pool_params *pool_params,
                                        const cmsis_nn_dims *input_dims,
                                        const q7_t *src,
                                        const cmsis_nn_dims *filter_dims,
                                        const cmsis_nn_dims *output_dims,
                                        q7_t *dst)
{
    const int32_t input_y = input_dims->h;
    const int32_t input_x = input_dims->w;
    const int32_t output_y = output_dims->h;
    const int32_t output_x = output_dims->w;
    const int32_t stride_y = pool_params->stride.h;
    const int32_t stride_x = pool_params->stride.w;
    const int32_t kernel_y = filter_dims->h;
    const int32_t kernel_x = filter_dims->w;
    const int32_t pad_y = pool_params->padding.h;
    const int32_t pad_x = pool_params->padding.w;
    const int32_t act_min = pool_params->activation.min;
    const int32_t act_max = pool_params->activation.max;
    const int32_t channel_in = input_dims->c;
    (void)ctx;
    q7_t *dst_base = dst;

    for (int i_y = 0, base_idx_y = -pad_y; i_y < output_y; base_idx_y += stride_y, i_y++)
    {
        for (int i_x = 0, base_idx_x = -pad_x; i_x < output_x; base_idx_x += 2 * stride_x, i_x += 2)
        {
            /* Condition for kernel start dimension: (base_idx_<x,y> + kernel_<x,y>_start) >= 0 */
            const int32_t ker_y_start = MAX(0, -base_idx_y);
            const int32_t ker_x_start = MAX(0, -base_idx_x);

            /* Condition for kernel end dimension: (base_idx_<x,y> + kernel_<x,y>_end) < dim_src_<width,height> */
            const int32_t kernel_y_end = MIN(2, input_y - base_idx_y);
            const int32_t kernel_x_end = MIN(2, input_x - base_idx_x);

            const q7_t *start1 = src + channel_in * (ker_x_start + base_idx_x + (ker_y_start + base_idx_y) * input_x);
            const q7_t *start2 = (ker_y_start + 1 < kernel_y_end) ? start1 + channel_in * input_x : start1;
            const q7_t *start3 = (ker_x_start + 1 < kernel_x_end) ? start1 + channel_in : start1;
            const q7_t *start4 = (ker_y_start + 1 < kernel_y_end && ker_x_start + 1 < kernel_x_end) ? start1 + channel_in * input_x + channel_in : start1;

            const q7_t *start5 = src + channel_in * (ker_x_start + base_idx_x + stride_x + (ker_y_start + base_idx_y) * input_x);
            const q7_t *start6 = (ker_y_start + 1 < kernel_y_end) ? start5 + channel_in * input_x : start5;
            const q7_t *start7 = (ker_x_start + 1 < kernel_x_end) ? start5 + channel_in : start5;
            const q7_t *start8 = (ker_y_start + 1 < kernel_y_end && ker_x_start + 1 < kernel_x_end) ? start5 + channel_in * input_x + channel_in : start5;

            compare_and_replace_if_larger_q7_2x2(start1, start2, start3, start4, dst, channel_in);
            dst += channel_in;

            if (i_x + 1 < output_x)
            {
                compare_and_replace_if_larger_q7_2x2(start5, start6, start7, start8, dst, channel_in);
                dst += channel_in;
            }
        }
    }

    clamp_output_2x2(dst_base, output_x * output_y * channel_in, act_min, act_max);

    return ARM_CMSIS_NN_SUCCESS;
}

/**
 * @} end of Pooling group
 */
