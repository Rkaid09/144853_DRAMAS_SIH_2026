#include <stdio.h>
#include <stdlib.h>
#include "prahari_logmel.h"

int main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr, "usage: %s in.pcm out_float.bin out_int8.bin\n", argv[0]);
        return 2;
    }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror("open input"); return 1; }
    fseek(f, 0, SEEK_END);
    long bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    int n = (int)(bytes / 2);

    int16_t *pcm = malloc((size_t)n * sizeof(int16_t));
    if (fread(pcm, sizeof(int16_t), (size_t)n, f) != (size_t)n) {
        fprintf(stderr, "short read\n"); return 1;
    }
    fclose(f);

    static float feats[PRAHARI_CTX_FRAMES * PRAHARI_MEL_BINS];
    static int8_t q[PRAHARI_CTX_FRAMES * PRAHARI_MEL_BINS];

    prahari_frontend_init();
    int frames = prahari_logmel_window(pcm, n, feats);
    prahari_quantise(feats, frames * PRAHARI_MEL_BINS, q);

    FILE *o1 = fopen(argv[2], "wb");
    fwrite(feats, sizeof(float), (size_t)(frames * PRAHARI_MEL_BINS), o1);
    fclose(o1);
    FILE *o2 = fopen(argv[3], "wb");
    fwrite(q, sizeof(int8_t), (size_t)(frames * PRAHARI_MEL_BINS), o2);
    fclose(o2);

    printf("%d\n", frames);
    free(pcm);
    return 0;
}
