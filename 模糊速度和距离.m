clc;
clear all;
close all;
%% ϵͳ����
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
 
N=ptr*fs;         %  ��������
%% �ο��ź�
t=0:1/fs:(N-1)/fs; 
ref=rectpuls(t-Tp/2,Tp).*exp(1j*pi*kr*(t-Tp/2).^2);%�����źŵĹ���Գ�  
for m=1:M
    tao=2*(R-m*ptr*v)/c;
    fd=2*v/lamda;

    s(m,:)=rectpuls((t-Tp/2-tao),Tp).*exp(1j*pi*kr.*(t-Tp/2-tao).^2).*exp(-1j*2*pi*f0*(2*R/c)).*exp(1j*2*pi.*fd*m*ptr);   
    %����ѹ��
    mai(m,:)=fft(s(m,:)).*conj(fft(ref)); 
    mai1(m,:)=ifft(mai(m,:));
end
Rx = t.*c/2;
figure(1)
ss2=mai1(20,:);

plot(Rx,10*log((abs(ss2)/max(abs(ss2)))));            % ������ѹ����Ĳ��Σ�����Ϊ����
xlabel('����/m');ylabel('����/dB');axis([1500,4500,-inf,+inf])

%% ��λ��fft�����ٶ�
for n=1:length(t)
    ss(:,n)=fft(mai1(:,n));
end
mm=0:M-1;
mm1=mm/ptr/64*lamda/2;
figure(2)
imagesc(t*c/2,mm1,abs(ss));
grid on;
xlabel('����/m');
ylabel('�ٶ�/m/s');
title('����-������ͼ');
[x y]=find(ss==max(max(ss))); 
vv = (x-1)/ptr/64*lamda/2;  %���ʵ���ٶ�ֵ
detav = lamda/2/Tp ;
vmax = 1/ptr*lamda/2;

% ��ʾ��Ϣ
fprintf('�ٶ�m/s     �ٶȷֱ���     ���ģ���ٶ�\n');
disp([vv;detav;vmax]')
    
