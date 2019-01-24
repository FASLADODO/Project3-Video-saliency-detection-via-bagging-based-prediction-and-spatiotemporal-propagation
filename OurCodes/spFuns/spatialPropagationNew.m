function SALS = spatialPropagationNew(TPSAL,CURINFOR,image,flow)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ��ʱ�򴫲��Ļ����Ͻ��п��򴫲�,����kd��Ѱ�ҽ���
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
% TPSAL(ȫ�ߴ�)
% ���߶��¸������������ֵ
% 
% V1:2016.08.27 10:19PM
% ʹ���µ��������������Զ�������� + motion + location��
% V2: 2016.08.31 15:12PM
% OR�ڽ��д��� CURINFOR.ORLabels{ss,1};
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1. ��ȡ����
% im_R im_G im_B im_L im_A im_B1 im_H im_S im_V Magn Ori im_Y im_X
FEA = prepaFea(image,flow);

%% 2. ��ʼ
knn = 30;
SPSCALENUM = length(CURINFOR.fea);
SALS = cell(SPSCALENUM,1);
for ss=1:SPSCALENUM
    %2.1 initial ----------------------------------------------------------
    tmpSPinfor = CURINFOR.spinfor{ss,1};% ���߶��µķָ��� 
    regionSal = TPSAL{ss,1};% ������ĳ�ʼ������ֵ
    regionFea = computeRegionFea(FEA,tmpSPinfor);% �����������
    tmpCurORlabels = CURINFOR.ORLabels{ss,1};

    %2.2 �ֲ�����
    LPSAL = localPropagation0(regionSal,regionFea,tmpSPinfor);
%     LPSAL = localPropagation(regionSal,regionFea,tmpSPinfor,tmpCurORlabels);
    
    %2.3 ���ڴ���(ֱ�ӵ���lu�Ĺ���,����δ�ܽ��оֲ���һ��)
    ISORLABEL       = tmpCurORlabels(:,1);
    regionFea_in_OR = regionFea(ISORLABEL==1,:);
    LPSAL_in_OR     = LPSAL(ISORLABEL==1,:);
    paramPropagate.lamna = 0.5;             
    paramPropagate.nclus = 8;
    paramPropagate.maxIter=200;   
    INPSAL = ...
        descendPropagationNew(regionFea,regionSal,paramPropagate,tmpSPinfor.spNum,size(regionFea,2));
%     INPSAL_in_OR = ...
%         descendPropagationNew(regionFea_in_OR,LPSAL_in_OR,paramPropagate,size(regionFea_in_OR,1),size(regionFea_in_OR,2));
    
%     index_in_OR = find(ISORLABEL==1);
%     INPSAL = zeros(length(ISORLABEL),1);
%     nn=1;
%     for sp=1:length(ISORLABEL)
%         if ismember(sp,index_in_OR)
%           INPSAL(sp,1) = INPSAL_in_OR(nn,1);
%           nn = nn + 1;
%         end
%     end
    
    
    %2.4 ȫ�ִ���
    GPSAL = globalPropagation(regionSal,regionFea);
%     GPSAL_in_OR = globalPropagation(LPSAL_in_OR,regionFea_in_OR);
%     nn=1;
%     GPSAL = zeros(length(ISORLABEL),1);
%     for sp=1:length(ISORLABEL)
%         if ismember(sp,index_in_OR)
%           GPSAL(sp,1) = GPSAL_in_OR(nn,1);
%           nn = nn + 1;
%         end
%     end
    
    % save 
    SALS{ss,1}.LPSAL = LPSAL;
    SALS{ss,1}.INPSAL = INPSAL;
    SALS{ss,1}.GPSAL = GPSAL;
    
    % clear ---------------------------------------
    clear LPSAL INPSAL GPSAL
    clear regionFea_in_OR LPSAL_in_OR GPSAL_in_OR
    clear tmpSPinfor regionSal regionFea
end
clear TPSAL CURINFOR FullResultCur
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. ��ȡ�������ڴ���
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
FEA = zeros(height,width,13);
FEA(:,:,1) = im_R;FEA(:,:,2) = im_G;FEA(:,:,3) = im_B;
FEA(:,:,4) = im_L;FEA(:,:,5) = im_A;FEA(:,:,6) = im_B1;
FEA(:,:,7) = im_H;FEA(:,:,8) = im_S;FEA(:,:,9) = im_V;
FEA(:,:,10) = Magn;FEA(:,:,11) = Ori;
FEA(:,:,12) = im_Y;
FEA(:,:,13) = im_X;
clear im_R im_G im_B im_L im_A im_B1 im_H im_S im_V Magn Ori im_Y im_X
clear image 
end

% 2. �����������ֵ
function regionFea = computeRegionFea(FEA,tmpSPinfor)
im_R = FEA(:,:,1);im_G = FEA(:,:,2);im_B = FEA(:,:,3);
im_L = FEA(:,:,4);im_A = FEA(:,:,5);im_B1 = FEA(:,:,6);
im_H = FEA(:,:,7);im_S = FEA(:,:,8);im_V = FEA(:,:,9);
Magn = FEA(:,:,10);Ori = FEA(:,:,11);
im_Y = FEA(:,:,12);
im_X = FEA(:,:,13);

[height,width] = size(im_R);
regionFea = zeros(tmpSPinfor.spNum,size(FEA,3));clear FEA
for sp=1:tmpSPinfor.spNum
    pixelList = tmpSPinfor.pixelList{sp,1};

    tmpfea = [mean(im_R(pixelList)),mean(im_G(pixelList)),mean(im_B(pixelList)), ...
            mean(im_L(pixelList)),mean(im_A(pixelList)),mean(im_B1(pixelList)), ...
            mean(im_H(pixelList)),mean(im_S(pixelList)),mean(im_V(pixelList)), ...
            mean(Magn(pixelList)),mean(Ori(pixelList)),mean(im_Y(pixelList))/height, ...
            mean(im_X(pixelList))/width];
    regionFea(sp,:) = tmpfea;

    clear tmpfea pixelList
    
end

clear FEA tmpSPinfor

end

% 3. �ֲ�����
% 2016.08.31 OR�Դ���
% regionSal,regionFea ȫ�ߴ�״̬
function result = localPropagation(regionSal,regionFea,tmpSPinfor,tmpCurORlabels)
result = zeros(size(regionSal,1),1);
ISORLABEL = tmpCurORlabels(:,1);
index_OR = find(ISORLABEL==1);
index_out_OR = find(ISORLABEL==0);
for sp=1:tmpSPinfor.spNum
    SIGN = ismember(sp,index_out_OR);
    
    if SIGN==0  % OR��  
    tmpfea = regionFea(sp,:);
    tmpsal = regionSal(sp,:);
    tmpadjmat = tmpSPinfor.adjmat(sp,:);
    
    % �޳�OR������ 2016.08.31 -----------------------------------
    adjindexs = find(tmpadjmat==1);
    adjindexsSign = ismember(adjindexs, index_OR);% adjindexλ��OR���� 1
    adjindexs1 = adjindexs(adjindexsSign==1);% ����OR�����򣬱��
    
    if ~isempty(adjindexs1) % ��ֹΪ��
    adjsetfea = regionFea(adjindexs1,:);
    adjsetsal = regionSal(adjindexs1,:);
    % -----------------------------------------------------------
    
    allfea = [tmpfea;adjsetfea];% �ֲ���һ��
    allfea = allfea./repmat((sqrt(sum(allfea.*allfea))+eps),[size(allfea,1),1]);
    
    tmpfea = allfea(1,:);adjsetfea=allfea(2:end,:);
    feadiff = repmat(tmpfea,[size(adjsetfea,1),1]) - adjsetfea;
    feadiff = sqrt(sum(feadiff.*feadiff,2));% size(adjsetfea,1)*1
    feadiff(feadiff==0) = feadiff(feadiff==0) + eps;
    feadiff = 1/(feadiff);
    if sum(feadiff)==0
        feadiff = feadiff/(sum(feadiff)+eps);
    else
        feadiff = feadiff/(sum(feadiff));
    end
    result(sp,1) = sum(sum(feadiff'.*adjsetsal)) + tmpsal;
    
    clear feadiff adjsetsal tmpsal allfea tmpfea adjsetfea
    end
    
    end
end

result = normalizeSal(result);

clear regionSal regionFea tmpSPinfor tmpCurORlabels
end

% 3.1 �ֲ�����
% 2016.08.31 OR�Դ���
% regionSal,regionFea ȫ�ߴ�״̬
function result = localPropagation0(regionSal,regionFea,tmpSPinfor)
result = zeros(size(regionSal,1),1);
% ISORLABEL = tmpCurORlabels(:,1);
% index_OR = find(ISORLABEL==1);
% index_out_OR = find(ISORLABEL==0);
for sp=1:tmpSPinfor.spNum
%     SIGN = ismember(sp,index_out_OR);
    
%     if SIGN==0  % OR��  
    tmpfea = regionFea(sp,:);
    tmpsal = regionSal(sp,:);
    tmpadjmat = tmpSPinfor.adjmat(sp,:);
    
    % �޳�OR������ 2016.08.31 -----------------------------------
    adjindexs1 = find(tmpadjmat==1);
%     adjindexsSign = ismember(adjindexs, index_OR);% adjindexλ��OR���� 1
%     adjindexs1 = adjindexs(adjindexsSign==1);% ����OR�����򣬱��
    
    if ~isempty(adjindexs1) % ��ֹΪ��
    adjsetfea = regionFea(adjindexs1,:);
    adjsetsal = regionSal(adjindexs1,:);
    % -----------------------------------------------------------
    
    allfea = [tmpfea;adjsetfea];% �ֲ���һ��
    allfea = allfea./repmat((sqrt(sum(allfea.*allfea))+eps),[size(allfea,1),1]);
    
    tmpfea = allfea(1,:);adjsetfea=allfea(2:end,:);
    feadiff = repmat(tmpfea,[size(adjsetfea,1),1]) - adjsetfea;
    feadiff = sqrt(sum(feadiff.*feadiff,2));% size(adjsetfea,1)*1
    feadiff(feadiff==0) = feadiff(feadiff==0) + eps;
    feadiff = 1/(feadiff);
    if sum(feadiff)==0
        feadiff = feadiff/(sum(feadiff)+eps);
    else
        feadiff = feadiff/(sum(feadiff));
    end
    result(sp,1) = sum(sum(feadiff'.*adjsetsal)) + tmpsal;
    
    clear feadiff adjsetsal tmpsal allfea tmpfea adjsetfea
    end
    
%     end
end

result = normalizeSal(result);

clear regionSal regionFea tmpSPinfor tmpCurORlabels
end

% 4. ȫ�ִ���(ȥ���ռ���룬��Ϊ�����а�����λ����Ϣ)
function result = globalPropagation(regionSal,regionFea)

%    kdNum = size(tmpfea,1);
    knn=round(size(regionFea,1)*2/3);
    kdtree = vl_kdtreebuild(regionFea') ;% ���� feaDim*sampleNum
    [indexs, distance] = vl_kdtreequery(kdtree,regionFea',regionFea', 'NumNeighbors', knn) ;
    distance1 = distance(2:end,:);% ������һ�У��������(knn-1)*sampleNum
    indexs1 = indexs(2:end,:);
%     meanD = mean(distance1);
%     dist = distance1./(repmat(meanD,[(knn-1),1])+eps);
    dist = distance1./(repmat(sum(distance1),[(knn-1),1])+eps);
    result = regionSal' + sum(regionSal(indexs1).*dist);
    result = normalizeSal(result);
    result = result';
%     meanD=mean(mean(distance));
%     tmpposDistM = tmpSPinfor.posDistM;
%     tmpposDistM = Dist2WeightMatrix(tmpposDistM, 0.25);
%     
%      cor_posWeight=zeros(knn,kdNum);
%      for k=1:kdNum
%          cor_posWeight(:,k)=tmpposDistM(indexs(:,k),k);
%      end    

%      dist=exp(-1/meanD*distance);
%      dist=dist.*cor_posWeight;
%      SALLOCAL=SALCLASS'+sum(SALCLASS(indexs).*dist)./(sum(dist)+eps);

clear indexs distance indexs1 distance1
clear regionSal regionFea kdtree
end







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % 4. ���ڴ���:���������׾���ķ�ʽ���о���
% function result = intraClassPropagation(regionSal,regionFea,tmpSPinfor)
% result = zeros(size(regionSal,1),1);
% 
% 
% 
% 
% 
% 
% 
% end