#ifndef OPT_KERNEL
#define OPT_KERNEL

void test_device(uint32_t *input[], uint8_t *kernel_bins);
void opt_2dhisto();
void pre_alloc_device(uint32_t *data);
void dealloc_device(uint8_t *kernel_bins);
/* Include below the function headers of any other functions that you implement */


#endif
