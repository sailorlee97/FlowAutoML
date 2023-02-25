#ifndef _EXAMPLE_FILE_H
#define _EXAMPLE_FILE_H

#include "embedia.h"



#define MAX_SAMPLE 32

#define SELECT_SAMPLE 0

#if SELECT_SAMPLE == 0
uint16_t sample_data_id = 1;

static fixed sample_data[]= {
  FL2FX(0.35642111589266406), FL2FX(0.12703601190954264), FL2FX(0.0), FL2FX(1.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.00510204081632653), FL2FX(0.08874728007460367), FL2FX(0.05017241379310344), FL2FX(0.14856843575418996), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0004825090470446321), FL2FX(2.5476141199805433e-05), 
  FL2FX(0.00011750421056754534), FL2FX(4.812699464054168e-05), FL2FX(0.0), FL2FX(0.0084991455078125), FL2FX(0.00012062726176115802), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0)
};

#endif
#if SELECT_SAMPLE == 1
uint16_t sample_data_id = 1;

static fixed sample_data[]= {
  FL2FX(0.010714285714020605), FL2FX(0.00011479591837320386), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.04382965495803543), FL2FX(0.0), FL2FX(0.03107541899441341), FL2FX(0.0), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(3.916807018918178e-05), FL2FX(5.033257958881563e-06), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), 
  FL2FX(0.0), FL2FX(0.0)
};

#endif
#if SELECT_SAMPLE == 2
uint16_t sample_data_id = 2;

static fixed sample_data[]= {
  FL2FX(0.013278800041099438), FL2FX(0.00017632653062124112), FL2FX(1.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(1.0), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.015306122448979591), FL2FX(0.003730183400683867), FL2FX(0.01655172413793103), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.00012062726176115802), FL2FX(2.101125047406634e-06), 
  FL2FX(7.833614037836356e-05), FL2FX(0.0), FL2FX(0.0), FL2FX(1.52587890625e-05), FL2FX(0.00012062726176115802), FL2FX(0.0), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0)
};

#endif
#if SELECT_SAMPLE == 3
uint16_t sample_data_id = 3;

static fixed sample_data[]= {
  FL2FX(0.7674896233197999), FL2FX(0.5890403219688286), FL2FX(0.0), FL2FX(1.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.00510204081632653), FL2FX(0.30480335125893687), FL2FX(0.12517945110344827), FL2FX(0.5310380087988827), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.005910735826296743), FL2FX(0.0007786419238181083), 
  FL2FX(0.002154243860404998), FL2FX(0.0024083291031345095), FL2FX(0.0), FL2FX(1.52587890625e-05), FL2FX(0.0018094089264173703), 
  FL2FX(0.0), FL2FX(0.007678778788181817), FL2FX(0.01673980674681754), FL2FX(0.020097772727272728), FL2FX(0.0011771727272727273), 
  FL2FX(0.2675), FL2FX(0.48194662480376765), FL2FX(0.5575), FL2FX(0.07596881666666667)
};

#endif
#if SELECT_SAMPLE == 4
uint16_t sample_data_id = 1;

static fixed sample_data[]= {
  FL2FX(0.1313444220468982), FL2FX(0.01725135719475772), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.00510204081632653), FL2FX(0.06285803102269194), FL2FX(0.0721839080689655), FL2FX(0.062281773743016765), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0014475271411338963), FL2FX(0.00010995887748094716), 
  FL2FX(0.0005875210528377267), FL2FX(8.070178772274146e-05), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0014475271411338963), FL2FX(0.0), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0)
};

#endif
#if SELECT_SAMPLE == 5
uint16_t sample_data_id = 6;

static fixed sample_data[]= {
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(1.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), 
  FL2FX(0.015306122448979591), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), 
  FL2FX(0.0), FL2FX(0.00012062726176115802), FL2FX(0.0), FL2FX(7.833614037836356e-05), FL2FX(0.0), FL2FX(0.0), FL2FX(1.52587890625e-05), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), 

};

#endif
#if SELECT_SAMPLE == 6
uint16_t sample_data_id = 6;

static fixed sample_data[]= {
  FL2FX(0.09386631106630958), FL2FX(0.008810884353510759), FL2FX(0.0), FL2FX(1.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.00510204081632653), FL2FX(0.026732981038234377), FL2FX(0.030344827586206893), FL2FX(0.029329608938547486), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0004825090470446321), FL2FX(1.5408250347648647e-05), 
  FL2FX(0.00011750421056754534), FL2FX(9.5009813156416e-06), FL2FX(0.0), FL2FX(0.00018310546875), FL2FX(0.00012062726176115802), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0)
};

#endif
#if SELECT_SAMPLE == 7
uint16_t sample_data_id = 6;

static fixed sample_data[]= {
  FL2FX(0.47892859715177943), FL2FX(0.22937260118517205), FL2FX(0.0), FL2FX(1.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.00510204081632653), FL2FX(0.14575346248057194), FL2FX(0.3503448275862069), FL2FX(0.010893854748603353), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0009650180940892642), FL2FX(0.0003557905080275233), 
  FL2FX(0.00035251263170263603), FL2FX(8.822339793095773e-06), FL2FX(0.0), FL2FX(0.007659912109375), FL2FX(0.0007237635705669482), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0)
};

#endif
#if SELECT_SAMPLE == 8
uint16_t sample_data_id = 0;

static fixed sample_data[]= {
  FL2FX(0.0006219791319650933), FL2FX(3.868580403258601e-07), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.00510204081632653), FL2FX(0.017290090836182777), FL2FX(0.019310344827586205), 
  FL2FX(0.018869929560055867), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0053075995174909525), 
  FL2FX(0.00010785775243354053), FL2FX(0.0017625631585131802), FL2FX(7.029595104370542e-05), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0053075995174909525), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0)
};

#endif
#if SELECT_SAMPLE == 9
uint16_t sample_data_id = 6;

static fixed sample_data[]= {
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(1.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), 
  FL2FX(0.015306122448979591), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), 
  FL2FX(0.0), FL2FX(0.00012062726176115802), FL2FX(0.0), FL2FX(7.833614037836356e-05), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0024871826171875), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), 

};

#endif
#if SELECT_SAMPLE == 10
uint16_t sample_data_id = 1;

static fixed sample_data[]= {
  FL2FX(0.8649447746505765), FL2FX(0.748129463303476), FL2FX(0.0), FL2FX(1.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.00510204081632653), FL2FX(0.3992388409698477), FL2FX(0.005442684063448276), FL2FX(0.8913823037709497), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.004463208685162846), FL2FX(2.5563688076780712e-05), 
  FL2FX(0.0014100505268105441), FL2FX(0.002670963372359745), FL2FX(0.0), FL2FX(0.0084991455078125), FL2FX(0.00012062726176115802), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0)
};

#endif
#if SELECT_SAMPLE == 11
uint16_t sample_data_id = 6;

static fixed sample_data[]= {
  FL2FX(0.030916865515388525), FL2FX(0.0009558525736712168), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.00510204081632653), FL2FX(0.04971879241529375), FL2FX(0.0611367942413793), 
  FL2FX(0.04922425309357542), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.01097708082026538), 
  FL2FX(0.0007062406565595548), FL2FX(0.003564294387215542), FL2FX(0.00036674918947580823), FL2FX(0.0), FL2FX(0.0), FL2FX(0.01097708082026538), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0)
};

#endif
#if SELECT_SAMPLE == 12
uint16_t sample_data_id = 5;

static fixed sample_data[]= {
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(1.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), 
  FL2FX(0.00510204081632653), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), 
  FL2FX(0.0), FL2FX(0.00024125452352231604), FL2FX(0.0), FL2FX(7.833614037836356e-05), FL2FX(0.0), FL2FX(0.0), FL2FX(1.52587890625e-05), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), 

};

#endif
#if SELECT_SAMPLE == 13
uint16_t sample_data_id = 0;

static fixed sample_data[]= {
  FL2FX(0.5355798742282416), FL2FX(0.28684580174932883), FL2FX(0.0), FL2FX(1.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.15635685421199874), FL2FX(0.16905172413793101), FL2FX(0.18156424581005587), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0009650180940892642), FL2FX(0.00017167942574851703), 
  FL2FX(0.0001958403509459089), FL2FX(8.822339793095773e-05), FL2FX(0.0), FL2FX(0.002349853515625), FL2FX(0.0004825090470446321), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0)
};

#endif
#if SELECT_SAMPLE == 14
uint16_t sample_data_id = 6;

static fixed sample_data[]= {
  FL2FX(0.14986170497856388), FL2FX(0.02245853061849278), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.00510204081632653), FL2FX(0.08666459434255518), FL2FX(0.17517241379310344), FL2FX(0.017780403952513967), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0028950542822677927), FL2FX(0.000533685762041285), 
  FL2FX(0.0009792017547295446), FL2FX(3.743839066044488e-05), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0028950542822677927), FL2FX(0.0), 
  FL2FX(0.3181818181818182), FL2FX(0.2998585572842999), FL2FX(0.45454545454545453), FL2FX(0.18181818181818182), FL2FX(0.041755208333333335), 
  FL2FX(0.00017545572653061226), FL2FX(0.04182106666666667), FL2FX(0.04168935)
};

#endif
#if SELECT_SAMPLE == 15
uint16_t sample_data_id = 1;

static fixed sample_data[]= {
  FL2FX(0.0), FL2FX(0.0), FL2FX(1.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(1.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(3.916807018918178e-05), FL2FX(0.0), FL2FX(0.0), FL2FX(1.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0)
};

#endif
#if SELECT_SAMPLE == 16
uint16_t sample_data_id = 5;

static fixed sample_data[]= {
  FL2FX(0.017151783388997564), FL2FX(0.0002941836734843971), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(1.0), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.00510204081632653), FL2FX(0.004818153559216661), FL2FX(0.0), FL2FX(0.010824022346368716), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.00024125452352231604), FL2FX(0.0), FL2FX(3.916807018918178e-05), 
  FL2FX(1.7531572665767242e-06), FL2FX(0.0), FL2FX(0.003265380859375), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0)
};

#endif
#if SELECT_SAMPLE == 17
uint16_t sample_data_id = 1;

static fixed sample_data[]= {
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), 
  FL2FX(0.0), FL2FX(0.34637417289400063), FL2FX(0.3586206896551724), FL2FX(0.36312849162011174), FL2FX(0.0), FL2FX(0.0), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0015681544028950543), FL2FX(0.0005918168883528685), FL2FX(0.0), 
  FL2FX(2.9407799310319242e-05), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0015681544028950543), FL2FX(0.0), FL2FX(0.06565350909090908), 
  FL2FX(0.0), FL2FX(0.06565350909090908), FL2FX(0.06565350909090908), FL2FX(0.04262785), FL2FX(0.0), FL2FX(0.04262785), FL2FX(0.04262785), 

};

#endif
#if SELECT_SAMPLE == 18
uint16_t sample_data_id = 7;

static fixed sample_data[]= {
  FL2FX(0.0), FL2FX(0.0), FL2FX(1.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(1.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), 
  FL2FX(0.01020408163265306), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), 
  FL2FX(0.0), FL2FX(0.00012062726176115802), FL2FX(0.0), FL2FX(3.916807018918178e-05), FL2FX(0.0), FL2FX(0.0), FL2FX(1.0), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), 

};

#endif
#if SELECT_SAMPLE == 19
uint16_t sample_data_id = 0;

static fixed sample_data[]= {
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(1.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), 
  FL2FX(0.00024125452352231604), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.002349853515625), FL2FX(0.0), FL2FX(0.0), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0)
};

#endif
#if SELECT_SAMPLE == 20
uint16_t sample_data_id = 8;

static fixed sample_data[]= {
  FL2FX(0.23927375238803245), FL2FX(0.057251928569247545), FL2FX(0.0), FL2FX(1.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.00510204081632653), FL2FX(0.05214485545539322), FL2FX(0.10051724137931034), FL2FX(0.015363128491620113), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0014475271411338963), FL2FX(0.00015311948782975842), 
  FL2FX(0.00043084877208099956), FL2FX(1.493011349600823e-05), FL2FX(0.0), FL2FX(0.2692718505859375), FL2FX(0.0007237635705669482), 
  FL2FX(0.0), FL2FX(0.018752109090909092), FL2FX(0.0), FL2FX(0.018752109090909092), FL2FX(0.018752109090909092), FL2FX(0.0588484), 
  FL2FX(0.0), FL2FX(0.0588484), FL2FX(0.0588484)
};

#endif
#if SELECT_SAMPLE == 21
uint16_t sample_data_id = 1;

static fixed sample_data[]= {
  FL2FX(0.08428571427670056), FL2FX(0.0071040816325053085), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.11159465340379235), FL2FX(0.0), FL2FX(0.06983240223463687), FL2FX(0.0), FL2FX(0.0), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(3.916807018918178e-05), FL2FX(1.1310692042430477e-05), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), 
  FL2FX(0.0), FL2FX(0.0)
};

#endif
#if SELECT_SAMPLE == 22
uint16_t sample_data_id = 6;

static fixed sample_data[]= {
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(1.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), 
  FL2FX(0.015306122448979591), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), 
  FL2FX(0.0), FL2FX(0.00012062726176115802), FL2FX(0.0), FL2FX(7.833614037836356e-05), FL2FX(0.0), FL2FX(0.0), FL2FX(1.52587890625e-05), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), 

};

#endif
#if SELECT_SAMPLE == 23
uint16_t sample_data_id = 8;

static fixed sample_data[]= {
  FL2FX(0.38289416643431035), FL2FX(0.14660794272686772), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(1.0), FL2FX(1.0), FL2FX(0.0), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.1789451870065278), FL2FX(0.057809330627586204), FL2FX(0.3720992694134079), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0020506634499396865), FL2FX(0.00012475429968976888), 
  FL2FX(0.00047001684227018136), FL2FX(0.0003917458188895796), FL2FX(0.0), FL2FX(0.00128173828125), FL2FX(0.0006031363088057901), 
  FL2FX(0.0), FL2FX(0.00010127454545454546), FL2FX(2.994011783592645e-05), FL2FX(0.00013154545454545455), FL2FX(8.03e-05), 
  FL2FX(0.18333333333333335), FL2FX(0.20408163265306123), FL2FX(0.3333333333333333), FL2FX(0.08324868333333334)
};

#endif
#if SELECT_SAMPLE == 24
uint16_t sample_data_id = 1;

static fixed sample_data[]= {
  FL2FX(0.5724741648844937), FL2FX(0.32772666950651663), FL2FX(0.0), FL2FX(1.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.00510204081632653), FL2FX(0.20119676717438606), FL2FX(0.42732758620689654), FL2FX(0.019291201117318437), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0009650180940892642), FL2FX(0.0004339698691664451), 
  FL2FX(0.00027417649132427244), FL2FX(1.2498314706885678e-05), FL2FX(0.0), FL2FX(0.0012054443359375), FL2FX(0.0006031363088057901), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0)
};

#endif
#if SELECT_SAMPLE == 25
uint16_t sample_data_id = 3;

static fixed sample_data[]= {
  FL2FX(0.7350218189885765), FL2FX(0.5402570744663395), FL2FX(0.0), FL2FX(1.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.2466498996580665), FL2FX(0.04210403273103448), FL2FX(0.5482391280726258), FL2FX(0.0), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0071170084439083235), FL2FX(0.0003153438508649456), 
  FL2FX(0.001958403509459089), FL2FX(0.0022643439934343693), FL2FX(0.0), FL2FX(0.00128173828125), FL2FX(0.0019300361881785283), 
  FL2FX(0.0), FL2FX(0.032492196972727275), FL2FX(0.04631541251768034), FL2FX(0.05888950909090909), FL2FX(0.00022705454545454546), 
  FL2FX(0.25), FL2FX(0.00020272860643642072), FL2FX(0.25), FL2FX(0.25)
};

#endif
#if SELECT_SAMPLE == 26
uint16_t sample_data_id = 8;

static fixed sample_data[]= {
  FL2FX(0.3282941419118608), FL2FX(0.10777704361774372), FL2FX(0.0), FL2FX(1.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.00510204081632653), FL2FX(0.08823318427106), FL2FX(0.16919540227586205), FL2FX(0.03721069433659218), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0007237635705669482), FL2FX(0.00012886900290760687), 
  FL2FX(0.00023500842113509065), FL2FX(2.1094440659132842e-05), FL2FX(0.0), FL2FX(0.003814697265625), FL2FX(0.0003618817852834741), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0)
};

#endif
#if SELECT_SAMPLE == 27
uint16_t sample_data_id = 8;

static fixed sample_data[]= {
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(1.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.00011750421056754534), FL2FX(0.0), FL2FX(0.0), FL2FX(1.52587890625e-05), FL2FX(0.0), FL2FX(0.0), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0)
};

#endif
#if SELECT_SAMPLE == 28
uint16_t sample_data_id = 1;

static fixed sample_data[]= {
  FL2FX(0.0), FL2FX(0.0), FL2FX(1.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(1.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), 
  FL2FX(0.00510204081632653), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), 
  FL2FX(0.0), FL2FX(0.00024125452352231604), FL2FX(0.0), FL2FX(7.833614037836356e-05), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0020294189453125), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), 

};

#endif
#if SELECT_SAMPLE == 29
uint16_t sample_data_id = 8;

static fixed sample_data[]= {
  FL2FX(0.02100655873835877), FL2FX(0.00044127551022659565), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(1.0), FL2FX(1.0), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.009636307118433322), FL2FX(0.0), FL2FX(0.005412011173184358), FL2FX(0.0), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.00011750421056754534), FL2FX(1.7531572665767242e-06), 
  FL2FX(0.0), FL2FX(0.002044677734375), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0)
};

#endif
#if SELECT_SAMPLE == 30
uint16_t sample_data_id = 1;

static fixed sample_data[]= {
  FL2FX(0.12661703630879886), FL2FX(0.016031873893675094), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.00510204081632653), FL2FX(0.056196023842088896), FL2FX(0.038773946358620685), FL2FX(0.07068036710893855), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0010856453558504221), FL2FX(4.429871974948986e-05), 
  FL2FX(0.0005091849124593631), FL2FX(8.013625312061993e-05), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0010856453558504221), FL2FX(0.0), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0)
};

#endif
#if SELECT_SAMPLE == 31
uint16_t sample_data_id = 8;

static fixed sample_data[]= {
  FL2FX(0.0), FL2FX(0.0), FL2FX(1.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(1.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), 
  FL2FX(0.00510204081632653), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), 
  FL2FX(0.0), FL2FX(0.00024125452352231604), FL2FX(0.0), FL2FX(7.833614037836356e-05), FL2FX(0.0), FL2FX(0.0), FL2FX(1.52587890625e-05), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), 

};

#endif
#if SELECT_SAMPLE == 32
uint16_t sample_data_id = 9;

static fixed sample_data[]= {
  FL2FX(0.5693944946680214), FL2FX(0.3242100906287862), FL2FX(0.0), FL2FX(1.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.00510204081632653), FL2FX(0.20069761603978858), FL2FX(0.23453848834482757), FL2FX(0.21448798114525142), 
  FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.0), FL2FX(0.011942098914354644), FL2FX(0.002947528254003606), 
  FL2FX(0.004230151580431632), FL2FX(0.00189335329444265), FL2FX(0.0), FL2FX(0.007659912109375), FL2FX(0.006634499396863691), 
  FL2FX(0.0), FL2FX(0.0013125363636363636), FL2FX(0.0), FL2FX(0.0013125363636363636), FL2FX(0.0013125363636363636), FL2FX(0.043427158333333334), 
  FL2FX(0.0), FL2FX(0.043427158333333334), FL2FX(0.043427158333333334)
};

#endif


#endif