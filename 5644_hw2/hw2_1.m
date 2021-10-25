clear all ; close all ; clc ;

n = 2; 
trainNum = [100;1000;10000]; validateNum = 20000; 

mu(:,1) = [0;4]; Sigma(: ,: ,1) = [1 0;0 3]; 
mu(:,2) = [3;2]; Sigma(: ,: ,2) = [2 0;0 2];
mu(:,3) = [3;2]; Sigma(: ,: ,3) = [2 0;0 2];

p = [0.6 ,0.4];

labelVal = (rand(1,validateNum) >= p(1));
NcVal = [length(find(labelVal==0)),length(find(labelVal==1))];

xVal = zeros(n, validateNum);
xVal(:, labelVal==0) = mvnrnd(mu(:, 1), Sigma(:,:,1), NcVal(1))';
xVal(:, labelVal==1) = mvnrnd(mu(:, 2), Sigma(:,:,2), NcVal(2))';

% Part 1

discriminantScore = log(evalGaussian(xVal,mu(:,2),Sigma(:,:,2)))- log(evalGaussian(xVal,mu(:,1),Sigma(:,:,1)));
tauMax = log(sort(discriminantScore(discriminantScore >= 0))); 

midTauMax = [tauMax(1)-1 tauMax(1:end-1) + diff(tauMax)./2 tauMax(length(tauMax))+1];

for i = 1:length(midTauMax)
    dec = (discriminantScore >= midTauMax(i));
    falseAlarmP(i) = sum(dec==1 & labelVal==0)/NcVal(1); 
    correctDetectionP(i) = sum( dec==1 & labelVal==1)/NcVal(2) ; 
    errorP(i) = falseAlarmP(i)*p(1)+(1-correctDetectionP(i))*p(2); 
end

[ minError , minIndex ] = min(errorP) ;
minDec = (discriminantScore >= midTauMax(minIndex));
minFalseAlarm = falseAlarmP(minIndex); 
minCorrectDetection = correctDetectionP(minIndex);

figure(1); 
plot(falseAlarmP,correctDetectionP,'-',minFalseAlarm,minCorrectDetection,'o'); 
title('Minimum Risk ROC Curve');
legend ( 'ROC Curve ' , ' Calculated Min Error ' ) ;
xlabel('P_{False Alarm}'); ylabel('P_{Correct Detection}');
                                
grid1 = linspace(floor(min(xVal(1,:)))-2,ceil(max(xVal(1,:)))+2); 
grid2 = linspace(floor(min(xVal(2,:)))-2,ceil(max(xVal(2,:)))+2); 
[h,v] = meshgrid(grid1 ,grid2);
dsGrid = log(evalGaussian([h(:)';v(:)'],mu(:,2),Sigma(:,:,2)))- log(evalGaussian([h(:)';v(:)'],mu(:,1),Sigma(:,:,1)))-midTauMax(minIndex);
dS = reshape(dsGrid ,length(grid1),length(grid2));

figure(2);
plotData(minDec, labelVal, NcVal, p, [1 1 1], xVal', [grid1;grid2;dS],'D');



%part2&3

for i = 1:length(trainNum)
    label_Val = (rand(1,trainNum(i)) >= p(1));
    Nc_Val = [length(find(label_Val==0)) ,length(find(label_Val==1))];
    
    x = zeros(n, trainNum(i));
    x(:, label_Val==0) = mvnrnd(mu(:, 1), Sigma(:,:,1), Nc_Val(1))';
    x(:, label_Val==1) = mvnrnd(mu(:, 2), Sigma(:,:,2), Nc_Val(2))';

    Linear = [ones(trainNum(i), 1) x']; 
    sigmaLinear = zeros(n+1, 1);
    Quadratic = [ones(trainNum(i), 1) x(1,:)' x(2,:)' (x(1,:).^2)' (x(1 ,:) .*x(2 ,:) )' (x(2 ,:) .^2)']; 
    sigmaQuadratic = zeros(6, 1); label_Val = double(label_Val)';
    
    [sigmsL, costL]=fminsearch(@(t)(cost_func(t, Linear, label_Val, trainNum(i))), sigmaLinear);
    [sigmaQ, costQ]=fminsearch(@(t)(cost_func(t, Quadratic, label_Val, trainNum(i))), sigmaQuadratic);

    point1 = [min(Linear(:,2))-2, max(Linear(:,2))+2]; point2 = (-1./sigmsL(3)).*(sigmsL(2).*point1 + sigmsL(1));

    % logistic-linear-function-based
    figure(3); 
    plotTrainingData(label_Val, i, Linear, [point1;point2], 'A'); 

    linearTest = [ones(validateNum, 1) xVal']; 

    figure(4);
    linearError(i)=plotData([ones(validateNum, 1) xVal']*sigmsL >= 0, labelVal',NcVal,p, [1,3,i], linearTest(:,2:3), [point1;point2],'C'); 

    % logistic-quadratic-function-based
    figure(5); 
    grid1 = linspace(min(Quadratic(:,2))-6, max(Quadratic(:,2))+6);
    grid2 = linspace(min(Quadratic(:,3))-6, max(Quadratic(:,3))+6); desScore = bound(grid1 ,grid2 ,sigmaQ);

    plotTrainingData(label_Val, i, Quadratic, [grid1;grid2;desScore], 'B') 

    quadraticTest = [ones(validateNum, 1) xVal(1,:)' xVal(2,:)' (xVal(1,:).^2)' (xVal(1,:).*xVal(2,:))' (xVal(2,:).^2)'];

    % Quadratic: plot all decisions and boundary countour
    figure(6);
    quadraticError(i) = plotData([ones(validateNum, 1) xVal(1,:)' xVal(2,:)' (xVal(1,:).^2)' (xVal(1,:).*xVal(2,:))' (xVal(2,:).^2)']*sigmaQ >= 0, labelVal', NcVal, ...
        p, [1,3,i], quadraticTest(:,2:3) ,[grid1;grid2;desScore],'D'); 
end

fprintf('%.2f%%\n\n',minError *100); % Minimum P(error)
fprintf('%.2f%%\t%.2f%%\n' ,[linearError(1);quadraticError(1)]); % Linear error and quadratic error for size 100
fprintf('%.2f%%\t%.2f%%\n' ,[linearError(2);quadraticError(2)]); % Linear error and quadratic error for size 1000
fprintf('%.2f%%\t%.2f%%\n' ,[linearError(2);quadraticError(3)]); % Linear error and quadratic error for size 10000


% Functions
function cost = cost_func(theta , x, label ,N)
    h = 1 ./ (1 + exp(-x*theta));  
    cost = (-1/N)*((sum(label' * log(h)))+(sum((1-label)' * log(1-h))));
end


function bound_func = bound(Grid1, Grid2, param)
    z = zeros(length(Grid1), length(Grid2)); 
    for i = 1:length(Grid1)
        for j = 1:length(Grid2)
            z(i,j) = [1 Grid1( i ) Grid2( j ) Grid1( i )^2 Grid1( i )*Grid2( j ) Grid2( j)^2]*param; 
        end
    end
    bound_func = z';
end

function plotTrainingData(label , pic , x, par, LQtype) 
    subplot(1,3,pic); hold on;
    plot(x(label==0,2),x(label==0,3), 'o' ,x(label==1,2),x(label==1,3), '+');
    if LQtype == 'A' plot(par(1,:), par(2,:)); 
    elseif LQtype == 'B' contour(par(1 ,:) , par(2 ,:) , par(3:end ,:) , [0 , 0]) ;
    end
end

function errorResult = plotData(dec , label , Nc, p, fig , x, bound, LQtype2 )
    tn = find(dec==0 & label==0); % tn: true negative
    fp = find(dec==1 & label==0); %pFA = length(fp)/Nc(1); % fp: false positive
    fn = find(dec==0 & label==1); %pMD = length(fn)/Nc(2); % fn: false negative
    tp = find(dec==1 & label==1); % tp: true positive
    errorResult = (length(fp)/Nc(1)*p(1) + length(fn)/Nc(2)*p(2))*100; 

    subplot(fig(1), fig(2),fig(3)); 
    plot(x(tn,1),x(tn,2),'og'); hold on; plot(x(fp,1),x(fp,2),'or'); hold on; plot(x(fn,1),x(fn,2),'+r'); hold on; plot(x(tp,1),x(tp,2),'+g'); hold on;

    if LQtype2 == 'C' plot(bound(1,:), bound(2,:)); 
    elseif LQtype2 == 'D' contour(bound(1 ,:) , bound(2 ,:) , bound(3:end ,:) , [0 , 0]) ; 
    end
end



