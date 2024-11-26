#include <cuda.h>
#include <cuda_runtime.h>

#define RDB_CHANNELS 3


/**
 * @brief Images are 2D. Hence, we have 16 X 16 blocks and # of blocks needed for the image.
 * 
 */
extern int m, n;

typedef uint32_t unsigned int


__global__ colorToGreyScaleConversion(uint32_t * pin, unsigned float * pout, int height, int width){
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    if (row < height && col < width){
        int greyOffset = row * width + col;
        int rgbOffset = greyOffset * RDB_CHANNELS;
        uint32_t r = pin[rgbOffset], g = pin[rgbOffset + 1], b = pin[rgbOffset + 2];
        pout[greyOffset] = 0.21f*r + 0.71f*g + 0.7f*b;
    }
}

dim3 dimGrid(ceil(m / 16.0), ceil(n / 16.0), 1);
dim3 dimBlock(16, 16, 1);
colorToGreyScaleConversion<<dimGrid, dimBlock>>>(d_Pin, d_Pout, m, n);



/**
 * @brief Blurring an image with Gaussian Convolution
 * Note that the area around the image is also reffered to as a kernal
 */
__constant__ float d_kernel[KERNEL_AREA] = {
    1.0f, 2.0f, 1.0f,
    2.0f, 4.0f, 2.0f,
    1.0f, 2.0f, 1.0f
};

__global__ void gaussian_blur(const unsigned int *d_in, unsigned int *d_out, int h, int w) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;

    float3 blur_color = make_float3(0.0f, 0.0f, 0.0f);
    float weight_sum = 0.0f;

    for (int dy = -KERNEL_RADIUS; dy <= KERNEL_RADIUS; ++dy) {
        for (int dx = -KERNEL_RADIUS; dx <= KERNEL_RADIUS; ++dx) {
            int nx = x + dx;
            int ny = y + dy;

            if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                unsigned int color = d_in[ny * w + nx];
                float r = static_cast<float>((color & 0x00FF0000) >> 16);
                float g = static_cast<float>((color & 0x0000FF00) >> 8);
                float b = static_cast<float>((color & 0x000000FF));

                int offset = (dy + KERNEL_RADIUS) * KERNEL_DIM + (dx + KERNEL_RADIUS);
                float weight = d_kernel[offset];

                blur_color.x += r * weight;
                blur_color.y += g * weight;
                blur_color.z += b * weight;
                weight_sum += weight;
            }
        }
    }

    blur_color /= weight_sum;

    unsigned int r_out = static_cast<unsigned int>(blur_color.x);
    unsigned int g_out = static_cast<unsigned int>(blur_color.y);
    unsigned int b_out = static_cast<unsigned int>(blur_color.z);

    d_out[y * w + x] = (r_out << 16) | (g_out << 8) | b_out;
}


// int main(){
//     dim3 block(32, 32);
//     dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

//     gaussian_blur<<<grid, block>>>(d_in, d_out, h, w);

//     cudaDeviceSynchronize();
// }
