/** 
 * @file filter_inora.cpp
 * Routines for INORA filtering of a color image
 */
#include <stdio.h>
#include "image.h"
#include "math.h"
/** 
 * @brief Implements the INORA Filter
 *
 * @param[in] in_img Image pointer { rgb }
 * @param[in] block_size Radius of the Block { positive-odd }
 * @param[in] alpha alpha nearest pixels in comptation window { positive }
 * @param[in] h smooting prameter spatial domain { positive }
 * @param[in] sigma2 smoothing prameter for peer group { positive }
 *
 * @return Pointer to the filtered image or NULL
 *
 * @author Kusnik Damian 
 * @date 25.02.2024
*/

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

float compute_ROAD(int* in_data, int width, float r, float g, float b, int window_pos, int alpha, float sigma) {
	float w, weights[9], r1, g1, b1;

	int f = 1;
	int a = 0;
	for (int i = -f; i <= f; i++)
		for (int j = -f; j <= f; j++)
		{
			//if (i == 0 && j == 0 && pixel_pos==window_pos)
			//	continue;

			r1 = (in_data[window_pos + i * width + j] & 0XFF0000) >> 16;
			g1 = (in_data[window_pos + i * width + j] & 0XFF00) >> 8;
			b1 = (in_data[window_pos + i * width + j] & 0XFF);
			weights[a] = (r - r1) * (r - r1) + (g - g1) * (g - g1) + (b - b1) * (b - b1);
			a++;
		}

	w = 0;

	//this is faster than sorting
	for (int i = 0; (i <= alpha) && (i < a); i++)
	{
		float min = weights[0];
		int tmp = 0;
		for (int j = 1; j < 9; j++)
		{
			if (weights[j] < min)
			{
				min = weights[j];
				tmp = j;
			}
		}
		w += min;
		weights[tmp] = +INFINITY;
	}

	w /= (float)(alpha);
	w = expf(-(w / sigma));
	return w;
}


float compute_weight_inora_with_partial(int* in_data, float* partial_road, int width, float r, float g, float b, int pos, int alpha, float sigma, float sigma2, float* central_pix, int iter) {
	float w, weights = 0, r1, g1, b1;

	int f = 1;
	int pos2;
	int a = 0;
	float sum = 0;

	for (int i = -f; i <= f; i++)
		for (int j = -f; j <= f; j++)
		{
			pos2 = pos + i * width + j;
			r1 = (in_data[pos2] & 0XFF0000) >> 16;
			g1 = (in_data[pos2] & 0XFF00) >> 8;
			b1 = (in_data[pos2] & 0XFF);
			sum += ((r - r1) * (r - r1) + (g - g1) * (g - g1) + (b - b1) * (b - b1)) * partial_road[pos2];
			weights += partial_road[pos2];
			a++;
		}

	if (iter)
	{
		r1 = central_pix[0];
		g1 = central_pix[1];
		b1 = central_pix[2];
		w = compute_ROAD(in_data, width, r1, g1, b1, pos, alpha, sigma2);

		sum  += ((r - r1) * (r - r1) + (g - g1) * (g - g1) + (b - b1) * (b - b1)) * w;
		weights += w;
		a++;
	}

	sum /= weights;
	return  expf(-(sum / sigma));

}


void precalculate_pixels_INORA(int* in_data, float* pixels_ROAD, const int width, const int height, const int alpha, const float sigma, int ic, int ir)
{
	int f = 1;

	if (ic >= width - f || ir >= height - f || ic < f || ir < f)
		return;
	int pos = ir * width + ic;
	float r = (in_data[pos] & 0XFF0000) >> 16;
	float g = (in_data[pos] & 0XFF00) >> 8;
	float b = (in_data[pos] & 0XFF);

	float w = compute_ROAD(in_data, width, r, g, b, pos, alpha, sigma);
	pixels_ROAD[pos] = w;
	return;
}

void denoise_pixel_inora(int* in_data, float* pixels_ROAD, int* out_data, const int width, const int height, const int radius, const int alpha, const float sigma, const float sigma2, const int iter, float ic, float ir)

{
	int f = 1;
	float wsum = 0.0, w, mx, my, r, g, b, last_ir, last_ic, last_r, last_g, last_b;
	int iter_count = 0;

	if (ic >= width - f || ir >= height - f || ic < f || ir < f)
		return;

	int pos = ir * width + ic;
	int out_pos = pos;

	//if we are in the image borders
	//if (ir >= height - r || ic >= width - r) return;
	//if (ir < r || ic < r) return;
	r = (in_data[pos] & 0XFF0000) >> 16;
	g = (in_data[pos] & 0XFF00) >> 8;
	b = (in_data[pos] & 0XFF);


	float diff = 0;

	float central_pix[3];

	// go through all pixels in block
	do {

		int istart = MAX((int)round(ir) - radius - 1, 1);
		int iend = MIN((int)round(ir) + radius + 1, height - 2);
		int jstart = MAX((int)round(ic) - radius - 1, 1);
		int jend = MIN((int)round(ic) + radius + 1, width - 2);

		wsum = w = 0.0;
		last_ir = ir;
		last_ic = ic;
		last_r = r;
		last_g = g;
		last_b = b;

		central_pix[0] = r;
		central_pix[1] = g;
		central_pix[2] = b;

		r = 0;
		g = 0;
		b = 0;

		wsum = 0;
		mx = 0, my = 0;
		pos = (int)round(ir) * width + (int)round(ic);
		for (int i = istart; i <= iend; i++) { // i = y
			for (int j = jstart; j <= jend; j++) { // j = x
				int q = i * width + j;
				w = compute_weight_inora_with_partial(in_data, pixels_ROAD, width,
					(in_data[q] & 0XFF0000) >> 16,
					(in_data[q] & 0XFF00) >> 8,
					(in_data[q] & 0XFF),
					pos, alpha, sigma, sigma2, central_pix, iter_count);
				w *= pixels_ROAD[q];
				r += ((in_data[q] & 0XFF0000) >> 16)* w;
				g += ((in_data[q] & 0XFF00) >> 8)* w;
				b += (in_data[q] & 0XFF) * w;
				wsum += w;
				//zastanowic sie co robimy z x i y
				mx += i * w;
				my += j * w;

			}
		}

		diff = 0;
		r = r / wsum;
		g = g / wsum;
		b = b / wsum;

		ir = mx / wsum;
		ic = my / wsum;

		if (ir < 0)
			ir = 0;
		if (ic < -0)
			ic = 0;
		diff = (last_r - r) * (last_r - r) + (last_g - g) * (last_g - g) + (last_b - b) * (last_b - b)
			+ (last_ir - ir) * (last_ir - ir) + (last_ic - ic) * (last_ic - ic);
		iter_count++;
	} while (iter_count < iter && diff >0);

	out_data[out_pos] = ((int)(r) << 16) |
		((int)(g) << 8) |
		((int)(b));
	return;
}

Image *
filter_inora ( const Image * in_img, const int r, int alpha, const float sigma, const float sigma2, const int iter)
{
 SET_FUNC_NAME ( "filter_inora" );

 byte*** in_data;
 byte*** out_data;
 int num_rows, num_cols;
 Image* out_img;
 if ( !is_rgb_img ( in_img ) )
  {
   ERROR_RET ( "Not a color image !", NULL );
  }

 if ( !IS_POS ( r ) )
  {
   ERROR ( "Window size ( %d ) must be positive !", r );
   return NULL;
  }

 if ( !IS_POS ( alpha ) )
  {
   ERROR ( "Alpha value ( %d ) must be positive !", alpha );
   return NULL;
  }

 if ( !IS_POS ( sigma ) )
  {
   ERROR ( "Sigma value ( %d ) must be positive !", sigma );
   return NULL;
  }

 if (!IS_POS(sigma2))
 {
	 ERROR("Sigma2 value ( %d ) must be positive !", sigma2);
	 return NULL;
 }


 num_rows = get_num_rows(in_img);
 num_cols = get_num_cols(in_img);

 in_data = (byte***)get_img_data_nd(in_img);
 out_img = alloc_img(PIX_RGB, num_rows, num_cols);
 out_data = (byte***)get_img_data_nd(out_img);


 //size_t size_b = size_t(num_rows * num_cols) * sizeof(byte);
 size_t size_i = size_t(num_rows * num_cols) * sizeof(int);
 size_t size_f = size_t(num_rows * num_cols) * sizeof(float);

 int* int_in_data = (int*)malloc(size_i);
 int* int_out_data = (int*)malloc(size_i);
 float* pixels_ROAD = (float*)malloc(size_f);

 for (int i = 0; i < num_rows; i++) {
	 for (int j = 0; j < num_cols; j++)
	 {
		 int_in_data[i * num_cols + j] = (((int)in_data[i][j][0]) << 16) | ((int)in_data[i][j][1] << 8) | ((int)in_data[i][j][2]);
		 pixels_ROAD[i * num_cols + j] = 0;
	 }
 }

#pragma omp parallel \
    shared(int_in_data, pixels_ROAD)
 {
#pragma omp for schedule(dynamic) nowait
 for (int ir = 0; ir < num_rows; ir++) 
 	 for (int ic = 0; ic < num_cols; ic++)
         precalculate_pixels_INORA(int_in_data, pixels_ROAD, num_cols, num_rows, alpha, 2 * sigma2 * sigma2, ic, ir);
 }

#pragma omp parallel \
    shared(int_in_data, pixels_ROAD, int_out_data)
 {
#pragma omp for schedule(dynamic) nowait
 for (int ir = 0; ir < num_rows; ir++) 
 	 for (int ic = 0; ic < num_cols; ic++)
        denoise_pixel_inora(int_in_data, pixels_ROAD, int_out_data, num_cols, num_rows, r, alpha, 2 * sigma * sigma, 2 * sigma2 * sigma2, iter, ic, ir);
 }

 for (int i = 0; i < num_rows; i++)
	 for (int j = 0; j < num_cols; j++)
	 {
		 out_data[i][j][0] = (int_out_data[i * num_cols + j] >> 16) & 0xFF;
		 out_data[i][j][1] = (int_out_data[i * num_cols + j] >> 8) & 0xFF;
		 out_data[i][j][2] = (int_out_data[i * num_cols + j]) & 0xFF;
	 }

 // Free device memory

 free(int_in_data);
 free(int_out_data);

 return out_img;
}
