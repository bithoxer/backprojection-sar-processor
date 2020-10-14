
// !!! Ensure parameters are doubles to avoid integer division within program

// SAR System Parameters
double fc= 5428760000;                              // carrier frequency (Hz)
double bw = 170000000;                               // transmit bandwidth (Hz)
double kr = 1597249139246.997;                       // chirp rate
double PRF = 307.292;                                // effective pulse repetition frequency (Hz)
double azBW = 0.1920;                                // 3dB antenna azimuth beamwidth (rad) (3dB to 3dB)
double f_adc = 24485000;                             // adc sample rate (Hz)
double CABLE_DELAY =-1.2;                           // system cable delay (m)

// Data
# define N_PULSES 3884                              // number of pulses - data matrix height
# define N_RANGE_BINS 1702                          // number of fast time samples - data matrix width
# define N_GEOMETRY 4                               // number of geomtry data sets available

// Range Compression
# define BATCH 3884                                 // number of transforms done in parallel if sufficient memory on GPU
# define BATCH_SPLIT 4                              // SHOULD BE A FACTOR OF BATCH! // factor by which BATCH is divided to deal with inadequate space problem
# define NFFT 16384 // 1702 // 16384 // 1702                                  // 16384 // size of transform
# define RANK 1                                     // number of dimensions for transform