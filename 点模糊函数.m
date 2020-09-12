%�������3�����Ͼ����㶯�ز��źţ�����RD����
close all;clear;

%��������
c = 3*1e8;  %����
fc = 15*1e9;  %��Ƶ15GHz
lamda = c/fc;  %����
v = 100;  %�״��ٶ�100m/s
B = 200*1e6;  %�����źŴ���200MHz
thetaaz = 0.02;  %�������
Daz = lamda / thetaaz;  %��λ���߳ߴ�1m
R0 = 10*1e3;  %�ο���б��10km
Tr = 0.5*1e-6;  %�������ʱ��0.5us
Kr = B/Tr;  %���Ե�Ƶ�źŵ�Ƶб��
H = 1000;  %�״�߶�
Yc = sqrt(R0^2-H^2);  %��������,����������λ��  
center = [0,Yc,0];  %������������
CR = Daz/2;  %���򣨷�λ�򣩷ֱ���
Fs = 240*1e6;  %����Ƶ��
Dsar = lamda*R0/(2*CR);  %�ϳɿ׾����� (��λ��ֱ��ʹ�ʽ��CR = (lamda/(2*Dsar))*R0��
Tsar = Dsar/v;  %һ���ϳɿ׾�ʱ��
Ka = -2*v^2/(lamda*R0);  %�����յ�Ƶб��
Ba = abs(Ka*Tsar);  %�����մ���
PRF = 500;  %�����ظ�Ƶ��500hz
X = 150;Y = 225;  %������С
Rmin = sqrt((Yc-Y/2)^2+H^2);  %������С����
Rmax = sqrt((Yc+Y/2)^2+H^2+(Dsar/2)^2);  %����������
Nfast = 480;%ceil(((2*(Rmax-Rmin)/c+Tr)*Fs));  %��ʱ��ά��������
tf = 2*R0/c+(-(Nfast/2):(Nfast/2)-1)/Fs;  %��ʱ���������
Nslow = 1000;%ceil((X+Dsar)/v*PRF);  %��ʱ��ά����λ�򣩲�������
ts = (-(Nslow/2):(Nslow/2)-1)/PRF;  %��ʱ���������
pos = [0,0,0,1];  %Ŀ��������ĵ�λ��,[x,y,z,rcs],������Ϊ������ϵ��
disp('Ŀ��λ�ã���λ��б�࣬�߶ȣ���');
Rpos(1:3) = pos(1:3)+center  %Ŀ�����λ��
Rpos(:,4) = pos(:,4);

%�ز��ź�
signal = zeros(Nfast,Nslow);
Xs = ts.*v-Rpos(1);  
Ys = 0-Rpos(2);
Zs = H-Rpos(3);
sigma = Rpos(4);  %����ϵ��
R = sqrt(Xs.^2+Ys^2+Zs^2);  %б��
tau = 2*R/c;  %ʱ��
Tfast = tf'*ones(1,Nslow)-ones(Nfast,1)*tau;   %���ǿ�ʱ�䣨������  ������ʱ�䣨��λ��
Phase = pi*Kr.*Tfast.^2-(4*pi/lamda)*ones(Nfast,1)*R;   %��λ�ӳ�
signal = signal+sigma*exp(j*Phase).*(abs(Tfast)<=Tr/2).*(ones(Nfast,1)*(abs(Xs)<=Dsar/2));  %�ز�
S = fftshift(fft(fftshift(signal)));

%��ά��ѹ(BP�㷨��
href =exp(j*pi*Kr*(tf-2*R0/c).^2).*(abs(tf-2*R0/c)<=Tr/2);  %������ο�����
Hf = (fftshift(fft(fftshift(href))).')*ones(1,Nslow);
ComF = S.*conj(Hf);  %������ƥ���˲�
Sr = fftshift(ifft(fftshift(ComF)));  %������IFFT
Coms = fftshift(fft(fftshift(Sr.'))).';  %��λ��FFT
hs = exp(j*pi*Ka*ts.^2).*(abs(ts)<Tsar/2);  %��λ��ο�����
Hs = ones(Nfast,1)*fftshift(fft(fftshift(hs)));
ComS = Coms.*conj(Hs);  %��λ��ƥ���˲�
Saz = fftshift(ifft(fftshift(ComS.'))).';

%RD�㷨,sinc��ֵ
Coms_rcmc = zeros(Nfast,Nslow);
N = 6;  %��ֵ����
Rp = sqrt(sum((Rpos(2:3)-[0,H]).^2));  %Ŀ�굽�״���������
h = waitbar(0,'Sinc��ֵ��......');  %����һ��������
for m = 1:Nslow  %��ʱ��
  for n = N/2+1:Nfast  %��ʱ��
      %����ƫ����
      deltaR = (lamda/v)^2*(Rp+(n-Nfast/2)*c/2/Fs)*((m-Nslow/2)/Nslow*PRF)^2/8;
      DU = deltaR/(c/2/Fs);  %ƫ�ƾ��뵥Ԫ
      deltaDU = DU-floor(DU);  %ƫ�ƾ��뵥ԪС������
      for k = -N/2:N/2-1
          if (n+floor(DU)+k)>Nfast %�����߽�
              Coms_rcmc(n,m) = Coms_rcmc(n,m)+Coms(Nfast,m)*sinc(DU-k);
          else
              Coms_rcmc(n,m) = Coms_rcmc(n,m)+Coms(n+floor(DU)+k,m)*sinc(deltaDU-k);
          end
      end
  end
  waitbar(m/Nslow);
end
close(h);  %�رս�����
ComS_rcmc = Coms_rcmc.*conj(Hs);  %��λ��ѹ��
Saz_rcmc = fftshift(ifft(fftshift(ComS_rcmc.'))).';

%��ͼ
rf = c*tf/2;  %����
az = v*ts;  %��λ
faz = (-Nslow/2:Nslow/2-1)/Nslow*PRF;  %������Ƶ��
figure(1);
[f,Rf] = meshgrid(faz,rf);
subplot(121);
mesh(f,Rf,abs(Coms));view(0,90);
title('(a) δRCMC');
xlabel('������/Hz');ylabel('б��R/m');
subplot(122);
mesh(f,Rf,abs(Coms_rcmc));view(0,90);
title('(b) RCMC');
xlabel('������/Hz');ylabel('б��R/m');
figure(2);
[Az,Rf] = meshgrid(az,rf);
mesh(Az,Rf,abs(S));view(0,90);
title('(c) bp');
xlabel('������/Hz');ylabel('б��R/m');
figure(3);
[Az,Rf] = meshgrid(az,rf);
mesh(Az,Rf,abs(Saz));view(0,90);
title('��ά��ѹ������');
xlabel('��λx/m');ylabel('б��R/m');
figure(4);
mesh(Az,Rf,abs(Saz_rcmc));view(0,90);
title('RD�㷨������');
xlabel('��λx/m');ylabel('б��R/m');