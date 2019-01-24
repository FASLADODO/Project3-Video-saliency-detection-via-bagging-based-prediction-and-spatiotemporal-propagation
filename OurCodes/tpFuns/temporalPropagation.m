function SALS = temporalPropagation(CURINFOR,PREINFOR,FullResultCur)
% 根据前述得到的当前帧各尺度下每一个区域的mapsets，即于下一帧的最匹配区域
% mapsets OR_CUR--->OR_PRE
% CURINFOR
% fea/out_OR/spinfor(mapsets，region_center_prediction)
%
% PREINFOR
% spsal/fea/out_OR/spinfor
%
% FullResultCur 各区域对应显著性值及标签
% FullResultCur{ss,1}.OG_Label 
% FullResultCur{ss,1}.SalValue
% FullResultCur{ss,1}.COEF 
% SALS 时域传播后的各区域的显著性图
% 
% V2: 2016.08.18
% 
% 
beta1 = 1;
beta2 = 1;
[height,width] = size(CURINFOR.spinfor{1,1}.idxcurrImage);
DIAGDIST = sqrt(height*height + width*width);
SPSCALENUM = length(CURINFOR.fea);
SALS = cell(SPSCALENUM,1);
for ss=1:SPSCALENUM
    tmpSPcur = CURINFOR.spinfor{ss,1};
    tmpSPpre = PREINFOR.spinfor{ss,1};
    tmpFullKCR = FullResultCur{ss,1};
    tmpFeapre = PREINFOR.fea{ss,1};
    tmpFeacur = CURINFOR.fea{ss,1};
    
    tmpSal = zeros(tmpSPcur.spNum,1);
    
    out_or_pre = PREINFOR.out_OR{ss,1};
    adjmatrx_pre = tmpSPpre.adjmat;% 排除OR外相邻区域
    adjmatrx_pre(out_or_pre,:) = 0;
    adjmatrx_pre(:,out_or_pre) = 0;
    tmp_MAPSET = tmpSPcur.mapsets;
    tmp_region_center_prediction = tmpSPcur.region_center_prediction;
    tmp_region_center_pre = tmpSPpre.region_center;
    
    tmpsalpre = PREINFOR.spsal{ss,1};
    for mm=1:tmpSPcur.spNum
        tmpfeacurmm = tmpFeacur(mm,:);
        tmp_predicte_location = tmp_region_center_prediction(mm,:);
        spIndex = tmp_MAPSET(mm);
        if spIndex==0 % 表示curFrame的mm区域为OR外部区域
            tmpSal(mm,1)=tmpFullKCR.FullValue(mm);
        else
            % 开始时域传播--------------------------------------------------
            % 1 寻找相关集(相关集亦需是OR_pre内区域)
            corrsets = adjmatrx_pre(spIndex,:);% 1,0,...等标识符

            % 2 计算距离权重
            [v,vid] = find(corrsets==1);
            if length(vid)==1 % revised in 2016.08.21 19:52PM 去除单邻域出现NAN情形
            WT = 1;
            else
            corrfea = tmpFeapre(vid,:);
            feadiff = repmat(tmpfeacurmm,length(vid),1) - corrfea;
            feadiff_vid = sqrt(sum((feadiff.*feadiff),2));

            tmp_dist = zeros(length(vid), 2);
            tmp_dist(:, 1) = tmp_predicte_location(1) - tmp_region_center_pre(vid, 1);
            tmp_dist(:, 2) = tmp_predicte_location(2) - tmp_region_center_pre(vid, 2);
            tmp_dist2 = sqrt(tmp_dist(:, 1) .^ 2 + tmp_dist(:, 2) .^ 2);
            
%             feadiff_vid = normalizeSal(feadiff_vid);
%             tmp_dist2 = normalizeSal(tmp_dist2);
%             WT = exp(-beta1*feadiff_vid).*exp(-beta2*tmp_dist2);

%             feadiff_vid = feadiff_vid./max(feadiff_vid);% 距离归一化  
%             tmp_dist2 = tmp_dist2./DIAGDIST;         
            % --- revised in 2016.08.18 15:19PM ---------------------------
            % 重新进行归一化
            feadiff_vid = feadiff_vid./sum(feadiff_vid);
            tmp_dist2   = tmp_dist2./sum(tmp_dist2);  
            WT = [1-feadiff_vid].*[1-tmp_dist2];
            WT = WT./sum(WT);
            end
            % 3 加权求和（pre的sal通过距离因素施加影响于cur的sal）
            tmpSal(mm,1)= sum(sum(WT.*tmpsalpre(vid))) + tmpFullKCR.FullValue(mm);
            
            clear corrsets feadiff tmp_dist tmp_dist2 feadiff_vid 
        end
    end
    tmpSal = normalizeSal(tmpSal);% 归一化
    SALS{ss,1} = tmpSal;
    
    clear tmpSal tmpSPcur tmpSPpre tmpFullKCR tmpFEApre tmpFeacur
    clear out_or_pre adjmatrx_pre tmp_MAPSET tmp_region_center_prediction tmp_region_center_pre
    
end





end