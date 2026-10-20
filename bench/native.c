/* usage: bench-native <silk|celt> <complexity> <passes> <pcm.raw>
 * Encodes the whole PCM file per pass (fresh encoder each pass, init untimed); prints one JSON line. */
#include <opus.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static int cmp(const void *a, const void *b) { double x = *(double *)a, y = *(double *)b; return (x > y) - (x < y); }
int main(int argc, char **argv) {
    int silk = !strcmp(argv[1], "silk"), cx = atoi(argv[2]), passes = atoi(argv[3]);
    int fs = silk ? 16000 : 48000, frame = fs / 50, bitrate = silk ? 24000 : 64000, warm = 3;
    FILE *f = fopen(argv[4], "rb"); fseek(f, 0, SEEK_END); long n = ftell(f) / 2; rewind(f);
    short *s = malloc(n * 2); if (fread(s, 2, n, f) != (size_t)n) return 1; fclose(f);
    int frames = n / frame; float *x = malloc((size_t)frames * frame * 4);
    for (long i = 0; i < (long)frames * frame; i++) x[i] = s[i] * (1.0f / 32768.0f);
    double *t = malloc(passes * sizeof *t); long bytes = 0; unsigned char out[1276];
    for (int p = -warm; p < passes; p++) {
        int err; OpusEncoder *e = opus_encoder_create(fs, 1, silk ? OPUS_APPLICATION_RESTRICTED_SILK : OPUS_APPLICATION_RESTRICTED_CELT, &err);
        if (err) return 2;
        if (silk) opus_encoder_ctl(e, OPUS_SET_BANDWIDTH(OPUS_BANDWIDTH_WIDEBAND));
        opus_encoder_ctl(e, OPUS_SET_BITRATE(bitrate)); opus_encoder_ctl(e, OPUS_SET_COMPLEXITY(cx));
        opus_encoder_ctl(e, OPUS_SET_VBR(1)); opus_encoder_ctl(e, OPUS_SET_VBR_CONSTRAINT(silk ? 0 : 1));
        opus_encoder_ctl(e, OPUS_SET_DTX(0)); opus_encoder_ctl(e, OPUS_SET_INBAND_FEC(0));
        long b = 0; double t0 = now();
        for (int i = 0; i < frames; i++) { int r = opus_encode_float(e, x + (size_t)i * frame, frame, out, sizeof out); if (r < 0) return 3; b += r; }
        double dt = now() - t0; opus_encoder_destroy(e);
        if (p >= 0) t[p] = dt; bytes = b;
    }
    qsort(t, passes, sizeof *t, cmp);
    printf("{\"frames\":%d,\"bytes\":%ld,\"ms_min\":%.3f,\"ms_med\":%.3f}\n", frames, bytes, t[0], t[passes / 2]);
    return 0;
}
