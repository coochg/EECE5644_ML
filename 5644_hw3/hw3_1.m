clear all; close all; clc;


dimensions=3; numLabels=4; Lx={'L0','L1','L2','L3'};

mu_Param=2.7; sigma_Param=0.4;

D.d100.N=100; D.d200.N=200; D.d500.N=500; D.d1k.N=1000; D.d2k.N=2000; D.d5k.N=5000; D.d100k.N=100000;
type_D=fieldnames(D);

prior=ones(1,numLabels)/numLabels; 

mu.L0=mu_Param*[1 1 0]'; 
sigma_Random=sigma_Param*rand(dimensions,dimensions); 
Sigma.L0(:,:,1)=sigma_Random*sigma_Random'+eye(dimensions);

mu.L1=mu_Param*[1 0 0]'; 
sigma_Random=sigma_Param*rand(dimensions,dimensions); 
Sigma.L1(:,:,1)=sigma_Random*sigma_Random'+eye(dimensions);

mu.L2=mu_Param*[0 1 0]'; 
sigma_Random=sigma_Param*rand(dimensions,dimensions); 
Sigma.L2(:,:,1)=sigma_Random*sigma_Random'+eye(dimensions);

mu.L3=mu_Param*[0 0 1]'; 
sigma_Random=sigma_Param*rand(dimensions,dimensions); 
Sigma.L3(:,:,1)=sigma_Random*sigma_Random'+eye(dimensions);


for index=1:length(type_D)
    D.(type_D{index}).x=zeros(dimensions,D.(type_D{index}).N);
    [D.(type_D{index}).x,D.(type_D{index}).labels,D.(type_D{index}).N_l,D.(type_D{index}).p_hat]=Generate_data(D.(type_D{index}).N,prior,mu,Sigma,Lx,dimensions);
end

figure;
for index=1:length(type_D)-1
    subplot(3,2,index); 
    Plot_data(D.(type_D{index}).x,D.(type_D{index}).labels,Lx); 
    legend 'show'; title([type_D{index}]);
end

figure; 
Plot_data(D.(type_D{index}).x,D.(type_D{index}).labels,Lx); 
legend 'show'; title([type_D{end}]);

for index=1:length(type_D)
    [D.(type_D{index}).opt.PFE, D.(type_D{index}).opt.decisions]=Optimal(ones(numLabels,numLabels)-eye(numLabels),D.(type_D{index}).x,mu,Sigma,prior,D.(type_D{index}).labels,Lx);
    opPFE(index)=D.(type_D{index}).opt.PFE;
    fprintf('(Opt-pFE of N=%1.0f)Error=%1.2f%%\n',D.(type_D{index}).N,100*D.(type_D{index}).opt.PFE);
end

for index=1:length(type_D)-1
    [D.(type_D{index}).net,D.(type_D{index}).minPFE,D.(type_D{index}).optM,DataValidation.(type_D{index}).stats]=kfoldMLP(15,10,D.(type_D{index}).x,D.(type_D{index}).labels,numLabels);
    DataValidation.(type_D{index}).yVal=D.(type_D{index}).net(D.d100k.x);
    [~,DataValidation.(type_D{index}).decisions]=max(DataValidation.(type_D{index}).yVal); 
    DataValidation.(type_D{index}).decisions=DataValidation.(type_D{index}).decisions-1;

    DataValidation.(type_D{index}).pFE=sum(DataValidation.(type_D{index}).decisions~=D.d100k.labels)/D.d100k.N;
    
    output_pFE(index,1)=D.(type_D{index}).N; 
    output_pFE(index,2)=DataValidation.(type_D{index}).pFE; 
    output_pFE(index,3)=D.(type_D{index}).optM;

    fprintf('(NN-pFE of N=%1.0f)Error=%1.2f%%\n',D.(type_D{index}).N,100*DataValidation.(type_D{index}).pFE);
end


for index=1:length(type_D)-1  
    [~,select]=min(DataValidation.(type_D{index}).stats.mPFE);
    M(index)=(DataValidation.(type_D{index}).stats.M(select)); N(index)=D.(type_D{index}).N;
end


for index=1:length(type_D)-1
    figure; xlabel('Perceptrons Number'); ylabel('pfe');
    stem(DataValidation.(type_D{index}).stats.M,DataValidation.(type_D{index}).stats.mPFE); 
    title(['Error Probability & Perceptrons Number of ' type_D{index}]);
end


figure,semilogx(N(1:end-1),M(1:end-1),'o','LineWidth',2) 
xlabel('Data Points Number'); ylabel('Perceptrons Optimal Number');
xlim([50 10^4]);ylim([0 10]);
title('Perceptrons Optimal Number & Data Points Number');


figure,semilogx(output_pFE(1:end-1,1),output_pFE(1:end-1,2),'o','LineWidth',2)
xlabel('Data Points Number'); ylabel('pfe'); xlim([90 10^4]);
hold all; semilogx(xlim,[opPFE(end) opPFE(end)],'r--','LineWidth',2) 
legend('NN pFE','Optimal pFE')
title('Error Probability & Training Data Points');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Sig,Lab,Ln,Phat]= Generate_data(N,p,mu,Sigma,Lx,d)

cum_p = [0,cumsum(p)]; Sig = zeros(d,N); Z = rand(1,N); Lab = zeros(1,N);

    for ind=1:length(Lx)
        Dist = find(cum_p(ind)<Z & Z<=cum_p(ind+1)); 
        Ln(ind)=length(Dist);
        Sig(:,Dist) = mvnrnd(mu.(Lx{ind}),Sigma.(Lx{ind}),Ln(ind))'; 
        Lab(Dist)=ind-1; Phat(ind)=Ln(ind)/N;
    end

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Plot_data(x,labels,Lx)

for ind=1:length(Lx)
    indexOfP=labels==ind-1; 
    plot3(x(1,indexOfP),x(2,indexOfP),x(3,indexOfP),'.','DisplayName',Lx{ind}); 
    hold all;
end
xlabel('x1'); ylabel('x2'); zlabel('x3');

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
invSigma = inv(Sigma);
C = (2*pi)^(-n/2) * det(invSigma)^(1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(invSigma*(x-repmat(mu,1,N))),1); 
g = C*exp(E);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [pfeMini,dec]=Optimal(Matrix,x,mu,Sigma,pn,labs,lex) 

for ind = 1:length(lex)
    px_l(ind,:) =evalGaussian(x,mu.(lex{ind}),Sigma.(lex{ind})); 
end

px = pn*px_l; 
[~,dec] = min(Matrix*px_l.*repmat(pn',1,length(x))./repmat(px,length(lex),1),[],1); 

dec=dec-1; indexF=(dec~=labs);
pfeMini=sum(indexF)/length(x);
gridStyle='ox+*v';

figure;
xlabel('x1'); ylabel('x2');
legend 'show'; title('X Vector with Incorrect Classifications'); 

for ind=1:length(lex)
    indClass=dec==ind-1; plot3(x(1,indClass & ~indexF),x(2,indClass & ~indexF),x(3,indClass & ~indexF),gridStyle(ind),'Color',[0.39 0.83 0.07],'DisplayName',['Class ' num2str(ind) ' Correct Classification']);
    hold on;
    plot3(x(1,indClass & indexF),x(2,indClass & indexF),x(3,indClass & indexF),['r' gridStyle(ind)],'DisplayName',['Class ' num2str(ind) ' Incorrect Classification']);
    hold on; 
end

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
function [net_Out,pfe_Out, m_Optimal, count]=kfoldMLP(pNum,k,x,tags,num_Tags)

y=zeros(num_Tags,length(x));
for ind=1:num_Tags
    y(ind,:)=(tags==ind-1);
end

ind_Per=[1:length(x)/k:length(x) length(x)];

for q=1:pNum
    for ind=1:k

        param.val=ind_Per(ind):ind_Per(ind+1); param.train=setdiff(1:length(x),param.val);
        Ne=patternnet(q); Ne.layers{1}.transferFcn = 'softmax';
        Ne=train(Ne,x(:,param.train),y(:,param.train));
        valueOfy=Ne(x(:,param.val)); [~,valueOfTag]=max(valueOfy); valueOfTag=valueOfTag-1;
        pfe(ind)=sum(valueOfTag~=tags(param.val))/length(x)/k;
    
    end
    
    averageProb(q)=mean(pfe);
    count.M=1:q;
    count.mPFE=averageProb;
end

[~,m_Optimal]=min(averageProb);

for ind=1:10
    NET(ind)={['net' num2str(ind)]}; 
    Last.(NET{ind})=patternnet(m_Optimal); Last.layers{1}.transferFcn = 'softmax'; Last.(NET{ind})=train(Ne,x,y);
    valueOfy=Last.(NET{ind})(x); [~,valueOfTag]=max(valueOfy); valueOfTag=valueOfTag-1;
    last_PFE(ind)=sum(valueOfTag~=tags)/length(x);

end
[minPFE,outInd]=min(last_PFE);
count.finalPFE=last_PFE;

pfe_Out=minPFE; 
net_Out=Last.(NET{outInd}); 
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



