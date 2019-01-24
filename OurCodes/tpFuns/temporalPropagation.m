function SALS = temporalPropagation(CURINFOR,PREINFOR,FullResultCur)
% ����ǰ���õ��ĵ�ǰ֡���߶���ÿһ�������mapsets��������һ֡����ƥ������
% mapsets OR_CUR--->OR_PRE
% CURINFOR
% fea/out_OR/spinfor(mapsets��region_center_prediction)
%
% PREINFOR
% spsal/fea/out_OR/spinfor
%
% FullResultCur �������Ӧ������ֵ����ǩ
% FullResultCur{ss,1}.OG_Label 
% FullResultCur{ss,1}.SalValue
% FullResultCur{ss,1}.COEF 
% SALS ʱ�򴫲���ĸ������������ͼ
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
    adjmatrx_pre = tmpSPpre.adjmat;% �ų�OR����������
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
        if spIndex==0 % ��ʾcurFrame��mm����ΪOR�ⲿ����
            tmpSal(mm,1)=tmpFullKCR.FullValue(mm);
        else
            % ��ʼʱ�򴫲�--------------------------------------------------
            % 1 Ѱ����ؼ�(��ؼ�������OR_pre������)
            corrsets = adjmatrx_pre(spIndex,:);% 1,0,...�ȱ�ʶ��

            % 2 �������Ȩ��
            [v,vid] = find(corrsets==1);
            if length(vid)==1 % revised in 2016.08.21 19:52PM ȥ�����������NAN����
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

%             feadiff_vid = feadiff_vid./max(feadiff_vid);% �����һ��  
%             tmp_dist2 = tmp_dist2./DIAGDIST;         
            % --- revised in 2016.08.18 15:19PM ---------------------------
            % ���½��й�һ��
            feadiff_vid = feadiff_vid./sum(feadiff_vid);
            tmp_dist2   = tmp_dist2./sum(tmp_dist2);  
            WT = [1-feadiff_vid].*[1-tmp_dist2];
            WT = WT./sum(WT);
            end
            % 3 ��Ȩ��ͣ�pre��salͨ����������ʩ��Ӱ����cur��sal��
            tmpSal(mm,1)= sum(sum(WT.*tmpsalpre(vid))) + tmpFullKCR.FullValue(mm);
            
            clear corrsets feadiff tmp_dist tmp_dist2 feadiff_vid 
        end
    end
    tmpSal = normalizeSal(tmpSal);% ��һ��
    SALS{ss,1} = tmpSal;
    
    clear tmpSal tmpSPcur tmpSPpre tmpFullKCR tmpFEApre tmpFeacur
    clear out_or_pre adjmatrx_pre tmp_MAPSET tmp_region_center_prediction tmp_region_center_pre
    
end





end