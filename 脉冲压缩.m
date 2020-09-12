close all;
clear all;
clc

c = 3e8;
f0 = 10e9;
fs =100e6;
band = 10e6;
Tp = 10e-6;    
Kr = band/Tp;
prt = 100e-6;                                        
t = 0:1/fs:prt-1/fs;

target_r = 3000;
N = prt*fs; 
f = (-(N-1)/2:(N-1)/2)*fs/N;
tau = 2*target_r/c;

Amp = 1;
sig_r = Amp*rectpuls(t-tau-Tp/2,Tp).*exp(1i*pi*Kr*(t-tau-Tp/2).^2).*exp(-2i*pi*f0*tau);
save echo sig_r 

Rx = t.*c/2;
figure (1)
subplot(2,1,1)
plot(Rx,real(sig_r));title('�ز��źŵ�I·����');xlabel('(a) ����/m');ylabel('����/dB')
axis([2000,6000,-inf,inf]);
subplot(2,1,2)
plot(Rx,imag(sig_r));title('�ز��źŵ�Q·����');xlabel('(b) ����/m');ylabel('��λ')
axis([2000,6000,-inf,inf]);
figure (2)
plot(f*1e-6,abs(fftshift(fft(sig_r))));
title('�ز��źŵķ�Ƶ����');
xlabel('f/MHz');ylabel('����/dB');
%axis([-20,20,-inf,inf]);

figure(3)
Sf=rectpuls(t-Tp/2,Tp).*exp(1i*pi*Kr*(t-Tp/2).^2);%�ο��ź�
Scomp = conv(sig_r,conj(Sf));
t1 = (0:2*N-2)/fs;
x1 = (t1-Tp)*c/2;
S = abs(Scomp)/max(abs(Scomp));
plot(x1,20*log10(S)); 
title('�ز��ź�����ѹ�����');
xlabel('����/m');ylabel('����/dB');


