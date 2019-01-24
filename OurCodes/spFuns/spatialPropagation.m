function SALS = spatialPropagation(TPSAL,CURINFOR,FullResultCur)
% 在时域传播的基础上进行空域传播,利用kd树寻找近邻
% CURINFOR
% fea/out_OR/spinfor(mapsets，region_center_prediction)
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
% 各尺度下各区域的显著性值
% 
% FullResultCur 各区域对应显著性值及标签
% FullResult{ss,1}.FullValue 所有区域的显著性值
% FullResult{ss,1}.FullLabel 所有区域的标签（object/background）
% FullResult{ss,1}.FullCOEF 所有区域的表征系数 eg. 200*600(样本*系数表征维数)
%
clusterNum = 5;
knn = 10;
phaw = 0.5;
[height,width] = size(CURINFOR.spinfor{1,1}.idxcurrImage);
SPSCALENUM = length(CURINFOR.fea);
SALS = cell(SPSCALENUM,1);
for ss=1:SPSCALENUM
    tmpSPinfor = CURINFOR.spinfor{ss,1};% 单尺度下的分割结果
    tmpSAL = TPSAL{ss,1};% 单尺度下各区域的显著性值
    tmp_out_or = CURINFOR.out_OR{ss,1};% OR外区域编号
    tmpfea = CURINFOR.fea{ss,1};
    tmpFULLresult = FullResultCur{ss,1};
    
    %------------------- 与OR区域中进行空域传播 --------------------
    tmpfea(tmp_out_or,:) = [];% 舍弃OR外区域
    tmpSAL(tmp_out_or) = [];
    
    % 全局传播（利用原始特征）
    WGlobal = sqrt(tmpfea*tmpfea');
    WGlobal = WGlobal./(2*mean(mean(WGlobal))+eps);
    WGlobal=exp(-WGlobal); 
    tmpSALG = (WGlobal*tmpSAL)./sum(WGlobal,2) + tmpSAL;
    tmpSALG = normalizeSal(tmpSALG);
    
    % 类内传播（利用谱聚类的结果）
    tmpCOEF = tmpFULLresult.FullCOEF;
    tmpCOEF(tmp_out_or,:) = [];% 样本*系数表征维数
    CKSym = constructGraph(tmpCOEF,size(tmpCOEF,1));
    CKSym(CKSym==inf) = 0;CKSym(isnan(CKSym)) = 0;
    Grps = SpectralClustering(CKSym,clusterNum);% compute the clustering result
    tmpClusterIndex = Grps(:,2);% 各区域所属的cluter index
%     [tmpClusterIndex,clusterDists] = MySpectralClustering(CKSym,param.clusterNum);
%     tmpInitSal = tmpSALG;% revised in 2016.08.18 16:51PM
%     tmpInitSal(tmp_out_or) = [];
    SALCLASS = cluster2saliency2(tmpfea',tmpClusterIndex,tmpSPinfor, tmpSALG, phaw, tmp_out_or);
    
    % 局部邻域传播（KD树搜寻近邻）
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
     
     SALLOCAL(isnan(SALLOCAL)) = 0;% revised in 2016.08.21 14:46PM 去除NAN无效数据
     
     SALS{ss,1} = normalizeSal(SALLOCAL);
     
     
end
clear TPSAL CURINFOR FullResultCur
end