clear all ; close all ; clc ;

filenames{1,1} = '37073.jpeg'; filenames{1,2} = '106024.jpeg';
valueOfK = [2,3];

for index = 1:2 
    im_Data = imread(filenames{1,index}); 
    [param_R,param_C,param_D] = size(im_Data); 
    param_N = param_R*param_C; 
    im_Data = double(im_Data);
    index_Row = [1:param_R]'*ones(1,param_C); 
    index_Column = ones(param_R,1)*[1:param_C];
    feaData = [index_Row(:)';index_Column(:)']; 
    for par_d = 1:param_D
        dataD = im_Data(:,:,par_d); 
        feaData = [feaData;dataD(:)'];
    end
    feaRange = max(feaData,[],2)-min(feaData,[],2);
    x = diag(feaRange.^(-1))*(feaData-repmat(min(feaData,[],2),1,param_N)); 

        x = x'; Opts = statset('MaxIter',3000, 'TolFun',1e-5);
        model_GMM = fitgmdist(x, 2,'Options',Opts,'RegularizationValue',1e-11);
        p = posterior(model_GMM, x);
        [~, index_One] = max(p, [], 2);
        figure
        imagesc(reshape(index_One,param_R, param_C)); colormap('gray');      
        
        x = x'; kFold = 10;
        split = ceil(linspace(0, size(x,2), kFold+1));

        for k = 1:kFold
            split_Data(k,:) = [split(k)+1,split(k+1)];
        end        
        
        val_Li = zeros(kFold, 3);

        for p = 1:10
        for k = 1:kFold
                [row, column] = size(x);
                param_N = column;
                ind_Val = [split_Data(k,1):split_Data(k,2)];

                if k == 1
                    ind_Train = [split_Data(k, 2)+1:param_N];
                elseif k == kFold
                    ind_Train = [1:split_Data(k, 1)-1];
                else
                    ind_Train = [1:split_Data(k-1, 2), split_Data(k+1, 2):param_N];
                end
                
               for M = 2:4
                    Opts = statset('MaxIter',3000, 'TolFun',1e-5);
                    model_GMM = fitgmdist([x(1,ind_Train);x(2, ind_Train);x(3, ind_Train);x(4, ind_Train);x(5, ind_Train)]', M,'Options',Opts,'RegularizationValue',1e-11);
                    mu = (model_GMM.mu)';
                    val_Li(kFold,M-1) = sum(log(evalGMM([x(1,ind_Val);x(2, ind_Val);x(3, ind_Val);x(4, ind_Val);x(5, ind_Val)], model_GMM.ComponentProportion, mu, model_GMM.Sigma)));
               end
        end
        end
         
        [val, ind_Opt] = max(sum(val_Li)/kFold,[], 2)
        
        x = x'; Opts = statset('MaxIter',3000, 'TolFun',1e-5);
        model_GMM = fitgmdist(x, ind_Opt+1, 'Options',Opts,'RegularizationValue',1e-11);
        p = posterior(model_GMM, x);
        [~, index_Two] = max(p, [], 2);
        
        figure
        imagesc(reshape(index_Two,param_R, param_C)); colormap('gray');

end

function param = evalGMM(x,alpha,mu,Sigma)
    param = zeros(1,size(x,2));
    for m = 1:length(alpha)
        param = param + alpha(m)*evalGaussian(x,mu(:,m),Sigma(:,:,m));
    end
end
