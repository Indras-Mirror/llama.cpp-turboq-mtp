#pragma once

// Lloyd-Max codebooks for the TurboQuant CPU path.

static const float turboq_codebook_2bit[4] = {
    -1.5104f, -0.4528f, 0.4528f, 1.5104f,
};

static const float turboq_codebook_3bit[8] = {
    -2.1520f, -1.3440f, -0.7560f, -0.2451f,
     0.2451f,  0.7560f,  1.3440f,  2.1520f,
};

static const float turboq_codebook_4bit[16] = {
    -2.7326f, -2.0690f, -1.6180f, -1.2562f,
    -0.9424f, -0.6568f, -0.3881f, -0.1284f,
     0.1284f,  0.3881f,  0.6568f,  0.9424f,
     1.2562f,  1.6180f,  2.0690f,  2.7326f,
};

static const float turboq_boundaries_2bit[3] = {
    -0.9816f, 0.0000f, 0.9816f,
};

static const float turboq_boundaries_3bit[7] = {
    -1.7480f, -1.0500f, -0.5006f, 0.0000f,
     0.5006f,  1.0500f,  1.7480f,
};

static const float turboq_boundaries_4bit[15] = {
    -2.4008f, -1.8435f, -1.4371f, -1.0993f,
    -0.7996f, -0.5225f, -0.2583f,  0.0000f,
     0.2583f,  0.5225f,  0.7996f,  1.0993f,
     1.4371f,  1.8435f,  2.4008f,
};

// FWHT sign arrays (seed=42) for 128-element rotation — used by TBQ4_0.
// Same arrays as dflash's turbo-wht.cu, matching CUDA __constant__ arrays.
static const float turboq_wht_signs1[128] = {
    -1, 1, 1,-1,-1, 1,-1, 1,-1,-1, 1, 1, 1, 1, 1, 1, 1,-1, 1,-1, 1,-1,-1, 1, 1, 1,-1, 1, 1,-1,-1,-1,
    -1, 1, 1,-1, 1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1, 1,-1, 1, 1, 1, 1,-1, 1,
    -1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1, 1,-1,-1, 1, 1, 1,-1,-1, 1, 1,-1, 1, 1,-1, 1,-1,
    -1, 1, 1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1, 1,-1, 1, 1,-1,-1,-1,-1,-1, 1, 1,-1, 1, 1,-1, 1};
static const float turboq_wht_signs2[128] = {
     1, 1, 1, 1,-1, 1, 1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1, 1, 1,
     1, 1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1, 1,-1, 1, 1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1,
     1,-1, 1,-1,-1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1, 1,-1, 1,-1, 1, 1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1,
     1,-1, 1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1, 1,-1};

// Lloyd-Max centroids for N(0, 1/sqrt(128)) — native TBQ4 FWHT domain
static const float turboq_fwht_centroids_4bit[16] = {
    -0.241556f, -0.182907f, -0.143047f, -0.111065f,
    -0.083317f, -0.058069f, -0.034311f, -0.011353f,
     0.011353f,  0.034311f,  0.058069f,  0.083317f,
     0.111065f,  0.143047f,  0.182907f,  0.241556f,
};
static const float turboq_fwht_midpoints_4bit[15] = {
    -0.212232f, -0.162977f, -0.127056f, -0.097191f, -0.070693f,
    -0.046190f, -0.022832f,  0.000000f,  0.022832f,  0.046190f,
     0.070693f,  0.097191f,  0.127056f,  0.162977f,  0.212232f,
};

// 3-bit FWHT centroids — Lloyd-Max for N(0, 1/sqrt(128)), 8 levels
static const float turboq_fwht_centroids_3bit[8] = {
    -1.646828f, -0.895384f, -0.491349f, -0.157976f,
     0.157976f,  0.491349f,  0.895384f,  1.646828f,
};
static const float turboq_fwht_midpoints_3bit[7] = {
    -1.271106f, -0.693367f, -0.324663f, 0.0f,
     0.324663f,  0.693367f,  1.271106f,
};
