function SALS = spatialPropagationNew5(TPSAL,CURINFOR,image,flow,beta,IMSAL_TPSAL1)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ��ʱ�򴫲��Ļ����Ͻ��п��򴫲�
% CURINFOR
% fea/ORLabels/spinfor(mapsets��region_center_prediction)
% 
% spinfor{ss,1}.adjcMatrix;
% spinfor{ss,1}.colDistM 
% spinfor{ss,1}.clipVal 
% spinfor{ss,1}.idxcurrImage 
% spinfor{ss,1}.adjmat
% spinfor{ss,1}.pixelList 
% spinfor{ss,1}.area 
% spinfor{ss,1}.spNum 
% spinfor{ss,1}.bdIds 
% spinfor{ss,1}.posDistM 
% spinfor{ss,1}.region_center
% 
% FEA{ss,1}.colorHist_rgb 
% FEA{ss,1}.colorHist_lab 
% FEA{ss,1}.colorHist_hsv 
% FEA{ss,1}.lbpHist   
% FEA{ss,1}.hogHist  
% FEA{ss,1}.regionCov   
% FEA{ss,1}.geoDist    
% FEA{ss,1}.flowHist  
%
% TPSAL(ȫ�ߴ�)
% ���߶��¸������������ֵ
% 
% V1: 2016.10.14 20:01PM
% ����CVPR2016 GRAB˼����д����Ż�
% 
% V2:2016.10.18 15:45PM
% ������µ� iterative propagation �����޸�
%
% copyright by xiaofei zhou,shanghai university,shanghai,china
% zxforchid@163.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1. ��ȡ���� %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% im_R im_G im_B im_L im_A im_B1 im_H im_S im_V Magn Ori im_Y im_X
FEA = prepaFea(image,flow);

%% 2. begin %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SPSCALENUM = length(CURINFOR.fea);
SALS = cell(SPSCALENUM,1);
% betaindexs  = beta(:,2);
% betavalues = beta(:,1);
% [value,index] = max(betavalues);
% FEA_ID = betaindexs(index(end));% ��ʹ�����������
    alpha=0.99;
    theta=10;
AL = 0.75;AH=1.25; % ѡȡȷ�����������ڴ�����ǰ�����ڱ�����������������ǰ������
% for iter=1:20
for ss=1:SPSCALENUM
    %% 1 initial ------------------------------------------------
    tmpSPinfor = CURINFOR.spinfor{ss,1};% ���߶��µķָ��� 
    regionSal  = TPSAL{ss,1};% ������ĳ�ʼ������ֵ
    regionFea  = computeRegionFea(FEA,tmpSPinfor);% �����������
    spNum      = tmpSPinfor.spNum;
    tmpORlabel = CURINFOR.ORLabels{ss,1};
    
    %% 2 constrcut new Graph ------------------------------------
    adjcMatrix = tmpSPinfor.adjcMatrix;
    bdIds      = tmpSPinfor.bdIds;
    FeaDist = GetDistanceMatrix(regionFea);
    pixelnums = [];
    for ii=1:spNum
        tmppixelnum = tmpSPinfor.pixelList{ii,1}; 
        pixelnums = [pixelnums;length(tmppixelnum)];
        clear tmppixelnum
    end
    
    %% 3 �������Ż� ---------------------------------------------------------
    % 3.1 TPGT &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    ISORlabel        = tmpORlabel(:,1);
    [index_out_OR,~] = find(ISORlabel==0);% OR������
    [index_in_OR,~]  = find(ISORlabel==1);% OR��������
    regionSal_in_OR  = regionSal(index_in_OR);
    regionSal_in_OR  = normalizeSal(regionSal_in_OR);
    pixelnums_in_OR = [];tmp_regionSal = [];
    for pp=1:length(index_in_OR)
        tmppixelnum = tmpSPinfor.pixelList{index_in_OR(pp),1}; 
%         tmp_regionSal = [tmp_regionSal;mean(mean(IMSAL_TPSAL1(tmppixelnum)))];
        pixelnums_in_OR = [pixelnums_in_OR;length(tmppixelnum)];
        clear tmppixelnum
    end
    meanSal0 = sum(pixelnums_in_OR.*regionSal_in_OR)/sum(pixelnums_in_OR);% OR�����е������Ծ�ֵ

    
    % 3.2 �������� Background Propagation BP &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    adjcMatrix_nn = LinkNNAndBoundary2(adjcMatrix, bdIds); 
    % 3.2.1 ��ʼ����GRABͼ�ṹ�������� W1
    adjcMatrix_nn(bdIds,:) = 1;
    adjcMatrix_nn(:,bdIds) = 1;% ����GRABͼ�ṹ
    adjcMatrix_nn(1:spNum+1:end) = 0;% �ٴν��Խ�������
    % �ڽ������Ȩ��
    W = SetSmoothnessMatrix(FeaDist, adjcMatrix_nn, theta);
    for i=1:size(W,1) % һ�����ڱ߽磬һ���������壬����Ա߽�Ԫ����Ŀ
        for j=1:size(W,2)
            SIGN_I = ismember(i,bdIds);
            SIGN_J = ismember(j,bdIds);
%             SIGN = SIGN_I*SIGN_J;
            if (SIGN_I==1 && SIGN_J==0) || (SIGN_I==0 && SIGN_J==1)
               W(i,j) = W(i,j)/length(bdIds);
            end
        end
    end
    % 3.2.2 ���ڽ�����Ĺ���Ȩ��
    adjcMatrix_nn0 = tril(adjcMatrix_nn, -1);  
    [clipVal, ~, ~] = EstimateDynamicParas(adjcMatrix_nn, W);% clipVal ���еģ������������Сֵ���ľ�ֵ
    edgeWeight = W(adjcMatrix_nn0 > 0);
    edgeWeight = max(0, edgeWeight - clipVal);
    W0 = graphallshortestpaths(sparse(adjcMatrix_nn0), 'directed', false, 'Weights', edgeWeight);
    W1 = W;
    for i=1:size(W1,1) % ��̾���
        for j=1:size(W1,2)
            if W1(i,j)==0 && i~=j
               W1(i,j) = W0(i,j);
            end
        end
    end

    D = diag(sum(W1));
    optAff =(D-alpha*W1)\eye(spNum);
    optAff(1:spNum+1:end) = 0;
    BP=optAff*(regionSal < AL*meanSal0);% Ѱ�ҿɿ��ı���
    BPP = 1 - normalizeSal(BP);
    
    % 3.3 TPR  �Ż� saliency optimization &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
%     meanSal_BPP = mean(BPP(:));
    meanSal_BPP = sum(pixelnums.*BPP)/sum(pixelnums);% ���ؼ�ƽ��ֵ
    fgWeight = BPP> meanSal_BPP;% ���ó���ǰ����
    bgWeight = BPP< meanSal_BPP;
    mu = 0.1;    %small coefficients for regularization term
    W2 = W1 + adjcMatrix_nn * mu; %add regularization term
    D2 = diag(sum(W2));
    bgLambda = 5;   %global weight for background term, bgLambda > 1 means we rely more on bg cue than fg cue.
    E_bg = diag(bgWeight * bgLambda); %background term
    E_fg = diag(fgWeight);            %foreground term
    spNum = length(bgWeight);
    TPR =(D2 - W2 + E_bg + E_fg) \ (E_fg * ones(spNum, 1));
    TPR = normalizeSal(TPR);
    
%     [~, ~, neiSigma] = EstimateDynamicParas(adjcMatrix_nn, W1);
%     TPR = SaliencyOptimization(adjcMatrix_nn, bdIds, FeaDist, neiSigma, bgWeight, fgWeight);
  
    % 3.4 FPP ǰ������ &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    meanSal_TPR = sum(pixelnums.*TPR)/sum(pixelnums);
    TPR_GT = (TPR>AH*meanSal_TPR);% ���ÿɿ���ǰ��
    FPP = optAff*TPR_GT;
    FPP = normalizeSal(FPP);
    GPSAL = FPP;
    

    
    %% 5 SAVE --------------------------------------------------------------
    SALS{ss,1}.GPSAL = GPSAL;
    
    clear SPP GPSAL
end

% end


end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. ��ȡ�������ڴ��������ؼ���������
% ��Ϊֻ��7ά���������� 2016.10.16 22:46PM
function FEA = prepaFea(image,flow)
image = double(image);
[height,width,dims] = size(image);

% apperance
im_R = image(:,:,1);
im_G = image(:,:,2);
im_B = image(:,:,3);

[im_L, im_A, im_B1] = ...
    rgb2lab_dong(double(im_R(:)),double(im_G(:)),double(im_B(:)));
im_L=reshape(im_L,size(im_R));
im_A=reshape(im_A,size(im_R));
im_B1=reshape(im_B1,size(im_R));
        
imgHSV=colorspace('HSV<-',uint8(image));      
im_H=imgHSV(:,:,1);
im_S=imgHSV(:,:,2);
im_V=imgHSV(:,:,3);

% motion
curFlow = double(flow);
Magn    = sqrt(curFlow(:,:,1).^2+curFlow(:,:,2).^2);    
Ori     = atan2(-curFlow(:,:,1),curFlow(:,:,2));
clear flow

% location x,y
im_Y = repmat([1:height]',[1,width]);
im_X = repmat([1:width], [height,1]);

%% 2 preparation
FEA = zeros(height,width,7);
FEA(:,:,1) = im_L;FEA(:,:,2) = im_A;FEA(:,:,3) = im_B1;
FEA(:,:,4) = Magn;FEA(:,:,5) = Ori;FEA(:,:,6) = im_Y;
FEA(:,:,7) = im_X;

% FEA = zeros(height,width,13);
% FEA(:,:,1) = im_R;FEA(:,:,2) = im_G;FEA(:,:,3) = im_B;
% FEA(:,:,4) = im_L;FEA(:,:,5) = im_A;FEA(:,:,6) = im_B1;
% FEA(:,:,7) = im_H;FEA(:,:,8) = im_S;FEA(:,:,9) = im_V;
% FEA(:,:,10) = Magn;FEA(:,:,11) = Ori;
% FEA(:,:,12) = im_Y;
% FEA(:,:,13) = im_X;
clear im_R im_G im_B im_L im_A im_B1 im_H im_S im_V Magn Ori im_Y im_X
clear image 
end

% 2. �����������ֵ (���򼶵�����)
% ��Ϊֻ��7ά���������� 2016.10.16 22:46PM
function regionFea = computeRegionFea(FEA,tmpSPinfor)
% im_R = FEA(:,:,1);im_G = FEA(:,:,2);im_B = FEA(:,:,3);
% im_L = FEA(:,:,4);im_A = FEA(:,:,5);im_B1 = FEA(:,:,6);
% im_H = FEA(:,:,7);im_S = FEA(:,:,8);im_V = FEA(:,:,9);
% Magn = FEA(:,:,10);Ori = FEA(:,:,11);
% im_Y = FEA(:,:,12);
% im_X = FEA(:,:,13);
im_L = FEA(:,:,1);im_A = FEA(:,:,2);im_B1 = FEA(:,:,3);
Magn = FEA(:,:,4);Ori  = FEA(:,:,5);im_Y  = FEA(:,:,6);
im_X = FEA(:,:,7);

[height,width] = size(im_L);
regionFea = zeros(tmpSPinfor.spNum,size(FEA,3));clear FEA
for sp=1:tmpSPinfor.spNum
    pixelList = tmpSPinfor.pixelList{sp,1};
    tmpfea = [mean(im_L(pixelList)),mean(im_A(pixelList)),mean(im_B1(pixelList)), ...
            mean(Magn(pixelList)),mean(Ori(pixelList)),mean(im_Y(pixelList))/height, ...
            mean(im_X(pixelList))/width];
        
%     tmpfea = [mean(im_R(pixelList)),mean(im_G(pixelList)),mean(im_B(pixelList)), ...
%             mean(im_L(pixelList)),mean(im_A(pixelList)),mean(im_B1(pixelList)), ...
%             mean(im_H(pixelList)),mean(im_S(pixelList)),mean(im_V(pixelList)), ...
%             mean(Magn(pixelList)),mean(Ori(pixelList)),mean(im_Y(pixelList))/height, ...
%             mean(im_X(pixelList))/width];
    regionFea(sp,:) = tmpfea;

    clear tmpfea pixelList
    
end

clear FEA tmpSPinfor

end

% 3 ������
function W = SetSmoothnessMatrix(colDistM, adjcMatrix_nn, theta)
allDists = colDistM(adjcMatrix_nn > 0);
maxVal = max(allDists);
minVal = min(allDists);

colDistM(adjcMatrix_nn == 0) = Inf;
colDistM = (colDistM - minVal) / (maxVal - minVal + eps);
W = exp(-colDistM * theta);
end

% 4 2-hop & bb
function adjcMatrix = LinkNNAndBoundary2(adjcMatrix, bdIds)
%link boundary SPs
adjcMatrix(bdIds, bdIds) = 1;

%link neighbor's neighbor
adjcMatrix = (adjcMatrix * adjcMatrix + adjcMatrix) > 0;
adjcMatrix = double(adjcMatrix);

spNum = size(adjcMatrix, 1);
adjcMatrix(1:spNum+1:end) = 0;  %diagnal elements set to be zero
end