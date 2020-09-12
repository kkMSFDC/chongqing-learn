clc;
clear all;
close all;
%% 系统参数
ptr=100e-6;     
M=64;          
v=60;           
f0=10e9;         
Tp=10e-6;      
B=10e6;        
R=3e3;        
fs=100e6;     
c=3e8;         
kr=B/Tp;        
lamda=c/f0;   
 
N=ptr*fs;         %  采样个数
%% 参考信号
t=0:1/fs:(N-1)/fs; 
ref=rectpuls(t-Tp/2,Tp).*exp(1j*pi*kr*(t-Tp/2).^2);%发射信号的共轭对称  
for m=1:M
    tao=2*(R-m*ptr*v)/c;
    fd=2*v/lamda;

    s(m,:)=rectpuls((t-Tp/2-tao),Tp).*exp(1j*pi*kr.*(t-Tp/2-tao).^2).*exp(-1j*2*pi*f0*(2*R/c)).*exp(1j*2*pi.*fd*m*ptr);   
    %脉冲压缩
    mai(m,:)=fft(s(m,:)).*conj(fft(ref)); 
    mai1(m,:)=ifft(mai(m,:));
end
Rx = t.*c/2;
figure(1)
ss2=mai1(20,:);

plot(Rx,10*log((abs(ss2)/max(abs(ss2)))));            % 看脉冲压缩后的波形，横轴为距离
xlabel('距离/m');ylabel('幅度/dB');axis([1500,4500,-inf,+inf])

%% 方位向fft，求速度
for n=1:length(t)
    ss(:,n)=fft(mai1(:,n));
end
mm=0:M-1;
mm1=mm/ptr/64*lamda/2;
figure(2)
imagesc(t*c/2,mm1,abs(ss));
grid on;
xlabel('距离/m');
ylabel('速度/m/s');
title('距离-多普勒图');
[x y]=find(ss==max(max(ss))); 
vv = (x-1)/ptr/64*lamda/2;  %输出实测速度值
detav = lamda/2/Tp ;
vmax = 1/ptr*lamda/2;

% 显示信息
fprintf('速度m/s     速度分辨率     最大不模糊速度\n');
disp([vv;detav;vmax]')
    
