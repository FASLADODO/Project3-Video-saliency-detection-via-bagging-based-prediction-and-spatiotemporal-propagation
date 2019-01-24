function SALS = spatialPropagation(TPSAL,CURINFOR,FullResultCur)
% ��ʱ�򴫲��Ļ����Ͻ��п��򴫲�,����kd��Ѱ�ҽ���
% CURINFOR
% fea/out_OR/spinfor(mapsets��region_center_prediction)
% spinfor.idxcurrImage = idxcurrImage;
% spinfor.adjmat = adjmat;
% spinfor.pixelList =pixelList;
% spinfor.area = area;
% spinfor.spNum = spNum;
% spinfor.bdIds = bdIds;
% spinfor.posDistM = posDistM;
% spinfor.region_center = region_center;
%
% TPSAL
% ���߶��¸������������ֵ
% 
% FullResultCur �������Ӧ������ֵ����ǩ
% FullResult{ss,1}.FullValue ���������������ֵ
% FullResult{ss,1}.FullLabel ��������ı�ǩ��object/background��
% FullResult{ss,1}.FullCOEF ��������ı���ϵ�� eg. 200*600(����*ϵ������ά��)
%
clusterNum = 5;
knn = 10;
phaw = 0.5;
[height,width] = size(CURINFOR.spinfor{1,1}.idxcurrImage);
SPSCALENUM = length(CURINFOR.fea);
SALS = cell(SPSCALENUM,1);
for ss=1:SPSCALENUM
    tmpSPinfor = CURINFOR.spinfor{ss,1};% ���߶��µķָ���
    tmpSAL = TPSAL{ss,1};% ���߶��¸������������ֵ
    tmp_out_or = CURINFOR.out_OR{ss,1};% OR��������
    tmpfea = CURINFOR.fea{ss,1};
    tmpFULLresult = FullResultCur{ss,1};
    
    %------------------- ��OR�����н��п��򴫲� --------------------
    tmpfea(tmp_out_or,:) = [];% ����OR������
    tmpSAL(tmp_out_or) = [];
    
    % ȫ�ִ���������ԭʼ������
    WGlobal = sqrt(tmpfea*tmpfea');
    WGlobal = WGlobal./(2*mean(mean(WGlobal))+eps);
    WGlobal=exp(-WGlobal); 
    tmpSALG = (WGlobal*tmpSAL)./sum(WGlobal,2) + tmpSAL;
    tmpSALG = normalizeSal(tmpSALG);
    
    % ���ڴ����������׾���Ľ����
    tmpCOEF = tmpFULLresult.FullCOEF;
    tmpCOEF(tmp_out_or,:) = [];% ����*ϵ������ά��
    CKSym = constructGraph(tmpCOEF,size(tmpCOEF,1));
    CKSym(CKSym==inf) = 0;CKSym(isnan(CKSym)) = 0;
    Grps = SpectralClustering(CKSym,clusterNum);% compute the clustering result
    tmpClusterIndex = Grps(:,2);% ������������cluter index
%     [tmpClusterIndex,clusterDists] = MySpectralClustering(CKSym,param.clusterNum);
%     tmpInitSal = tmpSALG;% revised in 2016.08.18 16:51PM
%     tmpInitSal(tmp_out_or) = [];
    SALCLASS = cluster2saliency2(tmpfea',tmpClusterIndex,tmpSPinfor, tmpSALG, phaw, tmp_out_or);
    
    % �ֲ����򴫲���KD����Ѱ���ڣ�
    kdNum = size(tmpfea,1);
    kdtree = vl_kdtreebuild(tmpfea') ;
    [indexs, distance] = vl_kdtreequery(kdtree,tmpfea',tmpfea', 'NumNeighbors', knn) ;
    meanD=mean(mean(distance));
    tmpposDistM = tmpSPinfor.posDistM;
    tmpposDistM(tmp_out_or,:) = [];
    tmpposDistM(:,tmp_out_or) = [];
    tmpposDistM = Dist2WeightMatrix(tmpposDistM, 0.25);
     cor_posWeight=zeros(knn,kdNum);
     for k=1:kdNum
         cor_posWeight(:,k)=tmpposDistM(indexs(:,k),k);
     end    
     dist=exp(-1/meanD*distance);
     dist=dist.*cor_posWeight;
     SALLOCAL=SALCLASS'+sum(SALCLASS(indexs).*dist)./(sum(dist)+eps);
     
     SALLOCAL(isnan(SALLOCAL)) = 0;% revised in 2016.08.21 14:46PM ȥ��NAN��Ч����
     
     SALS{ss,1} = normalizeSal(SALLOCAL);
     
     
end
clear TPSAL CURINFOR FullResultCur
end