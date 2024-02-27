#include "image.h"


int main(int argc, char** argv)
{
	float elapsed_time;
	clock_t start_time;
	Image* in_img;
	Image* noisy_img;
	Image* out_img;
	int iter;
	int r;
	float sigma;
	float sigma2 = 50;
	int alpha = 3;

	if (argc < 3)
	{
		printf("argc: %d\n", argc);
		fprintf(stderr, "Usage: %s <reference image { rgb }> <noisy image {rgb}> <block_radius> <h> <iter> {alpha} {sigma} \n", argv[0]);
		exit(EXIT_FAILURE);
	}
	if (argc > 5)
	{
		r = atoi(argv[3]);
		sigma = atof(argv[4]);
		iter = atoi(argv[5]);
        if ( argc > 6 )
		    alpha = atoi(argv[6]);
        if ( argc == 8 )
    		sigma2 = atof(argv[7]);
	}
	else {
		r = 2;
		alpha = 3;
		sigma = 50;
		sigma2 = 50;
		iter = 10;
	}

	printf("Testing INORA Filter...\n");
	/* Read the input image */
	in_img = read_img(argv[1]);
	noisy_img = read_img(argv[2]);

	/* Make sure it's an rgb image */
	if (is_gray_img(in_img))
	{
		fprintf(stderr, "Input image ( %s ) must not be grayscale !", argv[1]);
		exit(EXIT_FAILURE);
	}

	/* Start the timer */
	start_time = start_timer();
	#ifdef CUDA
	    out_img = CUDA_filter_inora(noisy_img, r, alpha, sigma, sigma2, iter);
	#else
		out_img = filter_inora(noisy_img, r, alpha, sigma, sigma2, iter);
	#endif

    elapsed_time = stop_timer(start_time);

    write_img(out_img, "out.png", FMT_PNG);

    printf("Used parameters: r, alpha, sigma, iter: %d, %d, %f, %d, %f\n\n=========== \n\n", r, alpha, sigma, iter,  elapsed_time);

    printf("Measures: \n \n");

	#ifdef CUDA
        printf("Prat: %f\n", calculate_prat(in_img, out_img));
	#endif
    calculate_snr(in_img, out_img, NULL);
    calculate_ssim(in_img, out_img, NULL);

	#ifdef CUDA
        printf("\n\nCUDA INORA time = %f\n", elapsed_time);
	#else
        printf("\n\nINORA time = %f\n", elapsed_time);
	#endif

	/* Calculate and print various error measures */

	free_img(in_img);
    free_img(out_img);
	free_img(noisy_img);
	return EXIT_SUCCESS;
}
