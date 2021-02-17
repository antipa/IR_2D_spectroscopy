load 2DIRdata_Nick
%%% this part of code only deal with the FFT of the regular sampled data
spec_orig=fft(data_2DIR,[],1); %FFT along t1 axis

%calculating w1 from t1
L=length(t1);
f=1/(1e-15*2*(t1(2)-t1(1)))*linspace(-1,1,L); 
f0=1719.60;
f_cm=f/2.9997e10+f0; 
w1=f_cm;
%plot spectrum
figure
contour(w3,w1,real(spec_orig),20)
axis([1930 2030 1930 2030])