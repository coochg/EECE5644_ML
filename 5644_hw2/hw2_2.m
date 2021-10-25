clear;clc;
% set original point
x0=0;
y0=0; 
r=1;
%plot with from K 1 to 4
for K=1:4 
    if K==1
        angle=2*pi;
    elseif K==2
        angle=[0,0.5]*2*pi;
    elseif K==3
        angle=[0,1/3,2/3]*2*pi;
    elseif K==4
        angle=[0,1/4,2/4,3/4]*2*pi;
    end
% set xi yi as reference 
xi = r*cos(angle); 
yi = r*sin(angle);    
rtrue=r*sqrt(rand(1,1));
seta=2*pi*rand(1,1);
% get the true position 
xtrue=x0+rtrue*cos(seta);
ytrue=y0+rtrue*sin(seta); 

x=linspace(-2,2);
y=linspace(-2,2);
[X,Y] = meshgrid(x,y);

%general n 
n=normrnd(0,0.3,1,K);
c=0;
sigmaX=0.25;
sigmaY=0.25;
sig=0.3;

for i=1:K
    ri=abs((xtrue-xi(i)).^2+((ytrue-yi(i)).^2)).^(0.5)+n(i);
    while (ri<=0)
          ri=abs((xture-xi(i)).^2+((ytrue-yi(i)).^2)).^(0.5)+normrnd(0,0.3,1,1);
    end     
     c=c-((ri-distance(xi(i),yi(i),X,Y)).^2)/(2*(sig^2));
end 
% get the MAP here
Z=-0.5*((X.^2/sigmaX^2)+(Y.^2/sigmaY^2))+c;
figure;
contour(X,Y,Z,'ShowText','on');hold on;
scatter(xi,yi,25,'r','filled'),hold on;
scatter(xtrue,ytrue,'k','+');hold on;
xlabel('x');ylabel('y');
title(['MAP estimation objective contours when K=',num2str(K)]);
end
function dis = distance(a,b,c,d)
dis=(abs((a-c).^2+(b-d).^2).^(0.5));
end
