function [TPSPSAL_Img,TPSPSAL_RegionSal] = SP16FUN(TPSAL1,CURINFOR,cur_image,MVF_Foward_f_fp,param)
switch param.sp16ID
    case '0'
        [TPSPSAL_Img,TPSPSAL_RegionSal] = spfun16_0(TPSAL1,CURINFOR,cur_image,MVF_Foward_f_fp,param);
    case '1'
        [TPSPSAL_Img,TPSPSAL_RegionSal] = spfun16_1(TPSAL1,CURINFOR,cur_image,MVF_Foward_f_fp,param);
    case '2'
        [TPSPSAL_Img,TPSPSAL_RegionSal] = spfun16_2(TPSAL1,CURINFOR,cur_image,MVF_Foward_f_fp,param);
    case '3'
        [TPSPSAL_Img,TPSPSAL_RegionSal] = spfun16_3(TPSAL1,CURINFOR,cur_image,MVF_Foward_f_fp,param);
    case '4'
        [TPSPSAL_Img,TPSPSAL_RegionSal] = spfun16_4(TPSAL1,CURINFOR,cur_image,MVF_Foward_f_fp,param);
    case '5'
        [TPSPSAL_Img,TPSPSAL_RegionSal] = spfun16_5(TPSAL1,CURINFOR,cur_image,MVF_Foward_f_fp,param);
    case '6'
        [TPSPSAL_Img,TPSPSAL_RegionSal] = spfun16_6(TPSAL1,CURINFOR,cur_image,MVF_Foward_f_fp,param);
    case '7'
        [TPSPSAL_Img,TPSPSAL_RegionSal] = spfun16_7(TPSAL1,CURINFOR,cur_image,MVF_Foward_f_fp,param);
end

clear TPSAL1 CURINFOR cur_image MVF_Foward_f_fp param
end