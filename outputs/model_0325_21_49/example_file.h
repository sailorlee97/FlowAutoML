#ifndef _EXAMPLE_FILE_H
#define _EXAMPLE_FILE_H

#include "embedia.h"



#define MAX_SAMPLE 32

#define SELECT_SAMPLE 0

#if SELECT_SAMPLE == 0
uint16_t sample_data_id = 1;

static float sample_data[]= {
  0.5, 0.296748, 0.0, 0.0, 5.630341, 0.359281, 1.721084, 5.251171, 3.333333, 1.721084, 0.344828, 0.046172, 0.0, 0.0, 0.0, 
  0.000862, 0.27027, 6.0, 0.063452, 0.026616, 0.063452, 0.00061, 0.0, 0.0, 0.038347, 0.039046, 0.017483, 0.035587, 1.497463, 
  0.46748, 0.0, 0.035714, 0.03529, 0.017007, 1.128617, 0.0, 0.03529, 0.033389, 0.001273, 0.0, 0.533333, 0.00346, 0.079646, 
  1.813857, 0.039046, 0.243902, 0.0, 0.714286, 0.016577
};

#endif
#if SELECT_SAMPLE == 1
uint16_t sample_data_id = 2;

static float sample_data[]= {
  0.5, 0.0, 0.114414, 1.211548, 1.149934, 0.0, 1.445732, 1.103468, 0.526316, 1.445732, 0.0, 0.885609, 1.447713, 1.112197, 
  0.014888, 0.02004, 0.0, 17.0, 0.885609, 0.757576, 0.885609, 1.337461, 1.129964, 0.757576, 0.169492, 0.757576, 1.184211, 
  0.885609, 0.437143, 1.2, 0.0, 1.184211, 0.815987, 1.184211, 0.0, 0.0, 0.815987, 0.757576, 1.328313, 0.0, 1.2, 0.0, 0.0, 
  0.017714, 0.757576, 1.0, 0.885609, 0.0, 1.004651
};

#endif
#if SELECT_SAMPLE == 2
uint16_t sample_data_id = 1;

static float sample_data[]= {
  0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01345, 0.0, 0.0, 0.00463, 0.0, 0.0, 6.0, 0.07177, 0.009145, 0.07177, 
  0.001541, 0.0, 0.0, 0.0, 0.0, 0.003008, 0.361446, 0.0, 0.0, 0.004918, 0.001239, 0.024058, 0.001642, 0.0, 0.004947, 0.024058, 
  0.0, 3e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.361446, 0.0, 0.0
};

#endif
#if SELECT_SAMPLE == 3
uint16_t sample_data_id = 1;

static float sample_data[]= {
  0.384615, 0.438103, 0.074828, 0.074828, 0.073511, 0.092593, 0.045733, 0.045808, 0.0, 0.045733, 0.0, 0.008187, 0.045738, 
  0.045786, 0.011758, 0.0, 0.0, 6.0, 0.035993, 0.008167, 0.035993, 0.004471, 0.071865, 0.0, 0.0, 0.0, 0.227273, 2.5, 0.0, 
  0.0, 0.014881, 0.146413, 0.048426, 0.172414, 0.0, 0.022573, 0.048426, 0.0, 0.002735, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.194175, 
  2.5, 0.645161, 0.004497
};

#endif
#if SELECT_SAMPLE == 4
uint16_t sample_data_id = 1;

static float sample_data[]= {
  0.588235, 0.004264, 0.070618, 4.978514, 0.009381, 0.000362, 0.000504, 0.00053, 0.0, 0.000504, 0.0, 0.008209, 0.000165, 
  0.000208, 0.065388, 0.0, 0.0, 6.0, 0.016942, 0.007584, 0.016942, 0.000643, 0.003012, 10.0, 0.000418, 0.029806, 0.001113, 
  0.0, 0.009129, 0.000137, 0.088852, 0.000714, 0.014722, 0.0008, 0.628802, 0.066874, 0.014722, 0.011046, 2e-06, 0.0, 0.000137, 
  6e-05, 0.008621, 5.5e-05, 0.029806, 0.0, 0.0, 0.769231, 0.0
};

#endif
#if SELECT_SAMPLE == 5
uint16_t sample_data_id = 2;

static float sample_data[]= {
  1.052632, 1.25, 0.0, 0.0, 0.011742, 0.57971, 0.018328, 0.072354, 0.0, 0.018328, 0.39267, 0.718927, 0.017848, 0.072194, 
  0.089463, 1.155817, 0.064935, 6.0, 0.078628, 0.808171, 0.078628, 0.055821, 0.008427, 0.0, 0.984153, 1.254868, 0.021882, 
  0.0, 1.458549, 0.682267, 0.068644, 0.678426, 0.884774, 0.467572, 1.021235, 0.035269, 0.884774, 0.90687, 0.57589, 1.25, 
  0.410959, 0.180596, 0.524073, 0.053851, 1.254868, 0.0, 0.0, 1.25, 0.0
};

#endif
#if SELECT_SAMPLE == 6
uint16_t sample_data_id = 18;

static float sample_data[]= {
  0.5, 0.0, 0.0, 0.0, 0.243369, 0.0, 1.804063, 0.528521, 2.5, 1.804063, 0.0, 1.24928, 1.804463, 0.52534, 0.096108, 9.995345, 
  0.0, 17.0, 0.547264, 4.393993, 0.547264, 0.63562, 0.327875, 3.081081, 9.597198, 8.611714, 0.681044, 0.361446, 2.559954, 
  8.979591, 0.134228, 0.681044, 2.478049, 0.681044, 0.0, 0.09901, 2.478049, 9.056743, 4.898757, 0.0, 8.979591, 10.0, 10.0, 
  0.150544, 8.611714, 0.16, 0.361446, 0.0, 0.146484
};

#endif
#if SELECT_SAMPLE == 7
uint16_t sample_data_id = 8;

static float sample_data[]= {
  0.246914, 0.0, 0.863994, 0.861328, 0.247484, 0.0, 0.047956, 0.04706, 0.0, 0.047956, 0.0, 0.440571, 0.058584, 0.047522, 
  0.082283, 0.0, 0.0, 17.0, 0.813817, 0.259571, 0.813817, 0.126174, 0.190017, 0.0, 0.0, 0.0, 0.122324, 1.326336, 1.597324, 
  0.0, 0.015083, 0.079286, 0.639669, 0.111111, 0.0, 0.029586, 0.639669, 0.0, 0.026171, 0.0, 0.0, 0.0, 0.0, 0.049099, 0.0, 
  0.215054, 1.328872, 0.0, 0.228957
};

#endif
#if SELECT_SAMPLE == 8
uint16_t sample_data_id = 12;

static float sample_data[]= {
  0.3125, 0.0, 0.0, 0.0, 0.001597, 0.0, 0.336078, 0.456631, 0.0, 0.336078, 0.0, 1.516517, 0.336078, 0.456631, 0.141311, 1.616633, 
  0.0, 17.0, 1.851538, 1.516517, 1.851538, 5.507995, 0.001597, 0.0, 1.416867, 0.0, 5.287212, 0.845771, 0.168938, 0.0, 0.5625, 
  5.114945, 1.851538, 5.282396, 0.0, 0.5625, 1.851538, 0.0, 5.507995, 0.0, 0.0, 1.616633, 1.416867, 0.001641, 0.0, 4.656627, 
  0.845771, 0.0, 4.630458
};

#endif
#if SELECT_SAMPLE == 9
uint16_t sample_data_id = 1;

static float sample_data[]= {
  0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.004606, 0.0, 0.0, 1e-06, 0.0, 0.0, 6.0, 0.004606, 0.004149, 0.004606, 
  0.000694, 0.0, 0.0, 0.0, 0.0, 0.140845, 0.004606, 0.0, 0.0, 8.6e-05, 0.140845, 0.006196, 0.140845, 0.0, 0.000118, 0.006196, 
  0.0, 0.000688, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0
};

#endif
#if SELECT_SAMPLE == 10
uint16_t sample_data_id = 2;

static float sample_data[]= {
  0.5, 0.0, 0.402023, 0.0, 0.397768, 0.0, 0.037035, 0.392617, 0.625, 0.037035, 0.0, 0.284091, 0.0, 0.0, 0.550829, 0.625, 
  0.0, 17.0, 0.284091, 0.357143, 0.284091, 0.013326, 0.0, 0.625, 0.625, 0.625, 0.059172, 0.284091, 0.0, 0.625, 0.576541, 
  0.059172, 0.325, 0.059172, 0.0, 0.526316, 0.325, 0.625, 0.033505, 0.0, 0.625, 0.0, 0.0, 0.0, 0.625, 0.0, 0.284091, 0.0, 
  0.0
};

#endif
#if SELECT_SAMPLE == 11
uint16_t sample_data_id = 2;

static float sample_data[]= {
  0.5, 0.0, 0.588378, 0.0, 0.013225, 0.0, 0.001722, 0.004574, 0.588235, 0.001722, 0.0, 0.09985, 0.0, 0.0, 0.115972, 0.001259, 
  0.0, 17.0, 0.150489, 0.138229, 0.150489, 0.000191, 0.0, 0.33264, 0.087336, 0.243902, 0.004983, 0.23753, 0.0, 0.005945, 
  0.255565, 0.004983, 0.18297, 0.004983, 0.0, 0.245098, 0.18297, 0.15073, 0.0003, 0.0, 0.005945, 0.0, 0.0, 0.0, 0.243902, 
  0.0, 0.23753, 0.0, 0.0
};

#endif
#if SELECT_SAMPLE == 12
uint16_t sample_data_id = 7;

static float sample_data[]= {
  0.57554, 0.229319, 0.000306, 0.000306, 2.147248, 0.401786, 1.555421, 1.851185, 0.0, 1.555421, 0.481928, 0.134313, 1.555421, 
  1.851185, 0.001114, 0.023057, 0.481928, 6.0, 0.104339, 0.134313, 0.104339, 0.082903, 2.147248, 0.0, 0.116901, 0.0, 0.30303, 
  0.0, 4.292361, 0.0, 0.0, 0.44686, 0.104339, 0.401786, 0.0, 0.0, 0.104339, 0.0, 0.082903, 0.5, 0.0, 0.023057, 0.116901, 
  2.291222, 0.0, 1.323529, 0.0, 0.5, 0.309413
};

#endif
#if SELECT_SAMPLE == 13
uint16_t sample_data_id = 20;

static float sample_data[]= {
  0.458716, 0.668379, 0.0, 0.0, 0.516936, 0.010018, 0.645969, 1.172539, 0.0, 0.645969, 0.053004, 0.812537, 0.645969, 1.172539, 
  0.013046, 0.015376, 0.053004, 6.0, 0.519365, 0.812537, 0.519365, 0.001015, 0.516936, 0.0, 0.302077, 0.0, 0.0092, 0.0, 4.418444, 
  0.0, 0.0, 0.010484, 0.519365, 0.010018, 0.0, 0.0, 0.519365, 0.0, 0.001015, 0.666667, 0.0, 0.015376, 0.302077, 0.598034, 
  0.0, 0.020003, 0.0, 0.666667, 0.00203
};

#endif
#if SELECT_SAMPLE == 14
uint16_t sample_data_id = 9;

static float sample_data[]= {
  0.776699, 0.5, 0.389253, 0.127571, 0.306109, 0.493827, 0.306109, 0.2833, 0.0, 0.306109, 0.0, 0.0, 0.33781, 0.314999, 0.0, 
  0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.242964, 0.0, 0.0, 0.0, 0.0, 0.0, 0.283616, 0.674157, 0.447386, 0.656566, 0.0, 0.508475, 
  0.143204, 0.449867, 0.0, 0.0, 0.0, 0.526316, 0.487805, 0.0, 0.0, 0.017405, 0.0, 0.0, 0.0, 0.5, 0.0
};

#endif
#if SELECT_SAMPLE == 15
uint16_t sample_data_id = 1;

static float sample_data[]= {
  0.695652, 0.428097, 0.00558, 0.094572, 0.007971, 0.043042, 0.000613, 0.002096, 0.0, 0.000613, 0.129032, 0.028816, 0.000667, 
  0.003137, 0.291592, 0.000873, 0.15873, 6.0, 0.02263, 0.028381, 0.02263, 0.000149, 0.017822, 3.04878, 0.032941, 0.206271, 
  0.001336, 0.0, 0.048539, 0.027073, 0.165119, 0.009126, 0.035131, 0.002644, 1.097488, 0.207729, 0.035131, 0.030952, 0.000249, 
  0.0, 0.016234, 0.001601, 0.045261, 0.00156, 0.206271, 0.0, 0.0, 0.0, 0.0
};

#endif
#if SELECT_SAMPLE == 16
uint16_t sample_data_id = 8;

static float sample_data[]= {
  0.37037, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.424687, 0.0, 0.0, 0.506744, 0.0, 0.0, 17.0, 0.667948, 0.232987, 
  0.667948, 0.039642, 0.0, 0.0, 0.0, 0.0, 0.038911, 0.88198, 0.0, 0.0, 0.181818, 0.030372, 0.538969, 0.038241, 0.0, 0.2, 
  0.538969, 0.0, 0.025189, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.885915, 0.0, 0.0
};

#endif
#if SELECT_SAMPLE == 17
uint16_t sample_data_id = 20;

static float sample_data[]= {
  0.714286, 0.602117, 0.0, 0.0, 0.057618, 0.998793, 0.64283, 0.08165, 0.0, 0.64283, 1.153996, 0.693692, 0.64283, 0.08165, 
  0.204708, 0.807777, 1.153996, 6.0, 0.980774, 0.693692, 0.980774, 1.220532, 0.057618, 0.0, 0.869361, 0.0, 0.324733, 0.128205, 
  0.382103, 0.0, 0.425532, 0.646658, 0.980774, 0.310856, 0.0, 0.425532, 0.980774, 0.0, 1.220532, 0.0, 0.0, 0.807777, 0.869361, 
  0.022121, 0.0, 0.11772, 0.128205, 0.0, 1.294037
};

#endif
#if SELECT_SAMPLE == 18
uint16_t sample_data_id = 18;

static float sample_data[]= {
  0.206186, 0.0, 2.684909, 6.090208, 0.333455, 0.0, 1.683801, 0.402984, 0.0, 1.683802, 0.0, 0.14256, 1.694664, 0.403865, 
  0.000643, 0.0, 0.0, 17.0, 0.426667, 0.072557, 0.426667, 4.48894, 0.143661, 0.0, 0.0, 0.0, 7.647059, 4.682927, 0.976693, 
  0.0, 0.000209, 2.713043, 0.271416, 5.064935, 0.0, 0.00047, 0.271416, 0.0, 1.207839, 0.0, 0.0, 0.0, 0.0, 0.160438, 0.0, 
  2.345679, 10.0, 0.0, 1.722964
};

#endif
#if SELECT_SAMPLE == 19
uint16_t sample_data_id = 18;

static float sample_data[]= {
  0.5, 0.0, 0.086364, 0.0, 4.643266, 0.0, 0.382158, 1.348307, 10.0, 0.382158, 0.0, 0.104167, 0.0, 0.0, 0.022561, 10.0, 0.0, 
  17.0, 0.104167, 0.165631, 0.104167, 0.002092, 0.0, 10.0, 10.0, 10.0, 0.010449, 0.104167, 42.210613, 10.0, 0.0, 0.010449, 
  0.145228, 0.010449, 0.0, 0.0, 0.145228, 10.0, 0.008779, 0.0, 10.0, 0.0, 0.0, 5.567703, 10.0, 0.032895, 0.104167, 0.0, 0.006431, 

};

#endif
#if SELECT_SAMPLE == 20
uint16_t sample_data_id = 19;

static float sample_data[]= {
  0.588235, 0.28997, 0.0, 0.0, 0.782228, 0.043162, 1.159715, 0.431478, 0.0, 1.159715, 0.326797, 0.697523, 1.159715, 0.431478, 
  0.000726, 0.005691, 0.326797, 6.0, 0.39796, 0.697523, 0.39796, 0.004673, 0.782228, 0.0, 0.180812, 0.0, 0.036248, 0.0, 3.269481, 
  0.0, 0.0, 0.046745, 0.39796, 0.04287, 0.0, 0.0, 0.39796, 0.0, 0.004673, 1.428571, 0.0, 0.005691, 0.180812, 0.648272, 0.0, 
  0.483871, 0.0, 0.0, 0.488948
};

#endif
#if SELECT_SAMPLE == 21
uint16_t sample_data_id = 2;

static float sample_data[]= {
  0.5, 0.768531, 0.427136, 0.427136, 0.663638, 0.085288, 0.611981, 0.477337, 0.0, 0.611981, 0.25641, 0.210229, 0.611981, 
  0.477337, 0.105613, 0.123234, 0.25641, 6.0, 0.066625, 0.210229, 0.066625, 0.00657, 0.663638, 0.0, 0.304714, 0.0, 0.066667, 
  0.0, 0.663761, 0.0, 0.043273, 0.094185, 0.066625, 0.085106, 0.0, 0.043273, 0.066625, 0.0, 0.00657, 0.526316, 0.0, 0.123234, 
  0.304714, 0.195562, 0.0, 0.0, 0.0, 0.454545, 0.0
};

#endif
#if SELECT_SAMPLE == 22
uint16_t sample_data_id = 2;

static float sample_data[]= {
  0.3125, 0.0, 0.0, 0.0, 0.045735, 0.0, 0.629552, 0.290867, 0.0, 0.629552, 0.0, 0.449311, 0.629552, 0.290867, 0.006456, 0.246222, 
  0.0, 17.0, 0.875736, 0.449311, 0.875736, 2.276244, 0.045735, 0.0, 0.38673, 0.0, 1.911544, 0.717489, 7.016224, 0.0, 0.007622, 
  1.739427, 0.875736, 1.897321, 0.0, 0.007622, 0.875736, 0.0, 2.276244, 0.0, 0.0, 0.246222, 0.38673, 0.067896, 0.0, 1.43339, 
  0.717489, 0.0, 1.761545
};

#endif
#if SELECT_SAMPLE == 23
uint16_t sample_data_id = 2;

static float sample_data[]= {
  0.5, 0.0, 0.178021, 0.0, 0.148699, 0.0, 0.016101, 0.017126, 0.434783, 0.016101, 0.0, 0.430108, 0.0, 0.0, 1.231263, 0.000124, 
  0.0, 17.0, 0.482655, 0.304507, 0.482655, 0.275862, 0.0, 0.549451, 0.030628, 0.356506, 0.344828, 0.508475, 0.0, 0.163934, 
  1.153675, 0.344828, 0.368421, 0.344828, 0.0, 1.156951, 0.368421, 0.314465, 0.040815, 0.0, 0.163934, 0.0, 0.0, 0.0, 0.356506, 
  0.0, 0.547945, 0.0, 0.0
};

#endif
#if SELECT_SAMPLE == 24
uint16_t sample_data_id = 17;

static float sample_data[]= {
  0.568182, 0.721325, 0.0, 0.0, 0.000739, 0.251451, 0.000495, 0.000377, 0.0, 0.000495, 0.134529, 0.29776, 0.000428, 0.000377, 
  4.788589, 0.832998, 0.151515, 6.0, 0.180424, 0.76065, 0.180424, 0.051561, 0.000645, 0.0, 0.754203, 1.061063, 0.194986, 
  0.0, 0.026021, 0.251333, 1.851852, 0.268176, 0.581089, 0.217391, 1.123338, 2.051756, 0.581089, 0.959143, 0.213458, 1.111111, 
  0.201342, 0.120164, 0.289679, 9.5e-05, 1.061063, 0.0, 0.0, 1.333333, 0.0
};

#endif
#if SELECT_SAMPLE == 25
uint16_t sample_data_id = 2;

static float sample_data[]= {
  0.5, 0.0, 0.226145, 0.0, 0.226145, 0.0, 0.226145, 0.226145, 0.5, 0.226145, 0.0, 0.5, 0.0, 0.0, 0.804327, 0.5, 0.0, 17.0, 
  0.5, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.5, 0.804938, 0.5, 0.5, 0.5, 0.0, 0.804369, 0.5, 0.5, 0.5, 0.0, 
  0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0
};

#endif
#if SELECT_SAMPLE == 26
uint16_t sample_data_id = 2;

static float sample_data[]= {
  0.434783, 0.0, 0.369445, 0.0, 0.053119, 0.0, 0.004076, 0.020277, 0.666667, 0.004076, 0.0, 0.088849, 0.0, 0.0, 0.341815, 
  0.001938, 0.0, 17.0, 0.178731, 0.132505, 0.178731, 0.010662, 0.0, 0.519481, 0.117188, 0.41721, 0.055249, 0.239234, 0.0, 
  0.16129, 0.651341, 0.045872, 0.228471, 0.053476, 0.0, 0.615942, 0.228471, 0.176991, 0.023997, 0.0, 0.294118, 0.0, 0.0, 
  0.0, 0.41721, 0.0, 0.239234, 0.0, 0.0
};

#endif
#if SELECT_SAMPLE == 27
uint16_t sample_data_id = 18;

static float sample_data[]= {
  0.471698, 0.073284, 0.0, 0.0, 0.003033, 0.082305, 0.001142, 0.001715, 0.0, 0.001142, 0.0, 0.012366, 0.001143, 0.001818, 
  0.110352, 3.8e-05, 0.0, 6.0, 0.038462, 0.007619, 0.038462, 0.016614, 0.001768, 0.645161, 0.006777, 0.014002, 0.136986, 
  0.0, 0.131914, 0.064475, 0.50397, 0.119205, 0.012674, 0.127119, 0.461256, 0.526316, 0.012674, 0.011184, 0.000802, 0.0, 
  0.075188, 0.000115, 0.01173, 0.000216, 0.014002, 0.0, 0.0, 0.0, 0.0
};

#endif
#if SELECT_SAMPLE == 28
uint16_t sample_data_id = 17;

static float sample_data[]= {
  0.531915, 0.027676, 0.134324, 0.134324, 0.036528, 0.27027, 0.006178, 0.00332, 0.0, 0.006178, 0.197368, 0.334116, 0.006178, 
  0.00332, 0.971076, 0.207073, 0.197368, 6.0, 0.416196, 0.334116, 0.416196, 0.053465, 0.036528, 0.0, 0.405302, 0.0, 0.015408, 
  0.0, 0.077688, 0.0, 0.411765, 0.041746, 0.416196, 0.017258, 0.0, 0.411765, 0.416196, 0.0, 0.053465, 0.555556, 0.0, 0.207073, 
  0.405302, 0.003994, 0.0, 0.0, 0.0, 0.555556, 0.0
};

#endif
#if SELECT_SAMPLE == 29
uint16_t sample_data_id = 11;

static float sample_data[]= {
  0.5, 0.528781, 0.0, 0.0, 0.002475, 0.212121, 0.012419, 0.015796, 0.0, 0.012419, 0.536398, 0.302892, 0.012419, 0.015796, 
  0.001533, 0.112545, 0.536398, 6.0, 0.125417, 0.302892, 0.125417, 0.026153, 0.002475, 0.0, 0.341828, 0.0, 0.20658, 0.0, 
  0.418724, 0.0, 0.000459, 0.215394, 0.125417, 0.212121, 0.0, 0.000459, 0.125417, 0.0, 0.026153, 0.769231, 0.0, 0.112545, 
  0.341828, 0.041405, 0.0, 0.0, 0.0, 0.526316, 0.0
};

#endif
#if SELECT_SAMPLE == 30
uint16_t sample_data_id = 8;

static float sample_data[]= {
  0.5, 0.0, 0.831872, 0.0, 0.013986, 0.0, 0.007043, 0.007052, 0.454545, 0.007043, 0.0, 0.601719, 0.0, 0.0, 0.250032, 0.231042, 
  0.0, 17.0, 0.601719, 0.458874, 0.601719, 0.601719, 0.0, 0.493482, 0.4, 0.475763, 0.5, 0.601719, 0.0, 0.454545, 0.215917, 
  0.5, 0.501355, 0.5, 0.0, 0.215322, 0.501355, 0.458874, 0.478655, 0.0, 0.454545, 0.0, 0.0, 0.0, 0.475763, 0.0, 0.601719, 
  0.0, 0.0
};

#endif
#if SELECT_SAMPLE == 31
uint16_t sample_data_id = 1;

static float sample_data[]= {
  1.012658, 1.472955, 10.0, 10.0, 3.742659, 0.017065, 0.889372, 0.943034, 1.428571, 0.889372, 0.0, 0.0, 0.010673, 0.006899, 
  0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.10047, 0.0, 0.0, 0.0, 0.0, 0.0, 3.242997, 0.027978, 0.0, 0.0248, 0.0, 0.02453, 
  0.825926, 0.0, 0.0, 0.0, 0.0, 1.666667, 0.017045, 0.0, 0.0, 3.586915, 0.0, 0.028846, 0.0, 2.857143, 0.0
};

#endif
#if SELECT_SAMPLE == 32
uint16_t sample_data_id = 2;

static float sample_data[]= {
  0.724638, 1.666667, 0.0, 0.0, 0.021097, 7.03268, 0.700299, 1.274197, 1.25, 0.700299, 4.203297, 0.385046, 0.700895, 1.273365, 
  0.005999, 3.090974, 0.097087, 6.0, 0.02675, 1.646258, 0.02675, 0.461283, 0.034376, 0.0, 2.355401, 3.217391, 5.982456, 0.0, 
  0.222044, 7.352764, 0.089321, 5.358896, 2.684729, 5.500795, 2.558199, 0.029248, 2.684729, 2.136236, 8.73293, 1.538462, 
  7.436419, 0.002925, 0.082816, 0.005299, 3.217391, 8.277513, 0.0, 2.0, 1.728292
};

#endif


#endif