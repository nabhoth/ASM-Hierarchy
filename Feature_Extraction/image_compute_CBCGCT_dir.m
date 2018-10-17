function [] = image_compute_CBCGCT_dir()
clear all;
VOCinit;
%f nargin < 5, methods = {@gc_segment, @ncut_segment, @roi_sub_i_segment, @salience_segment, @pbCGTG, @pbBGTG, @globalPb, @contours2ucm,@globalPbsc, @contours2ucm, @null};end;
methods = {@pbCG,@pbBG,@pbTG,@pbGM,@pbGM2};
%if nargin < 5, methods = {@gc_segment, @ncut_segment, @roi_sub_i_segment, @salience_segment, @pbCGTG, @pbBGTG, @globalPb,@globalPbsc};end;
%if nargin < 5, methods_name = {'globalPb','globalPbsc'};end;
methods_name = {'pbCG', 'pbBG', 'pbTG', 'pbGM', 'pbGM2'};

%path - change depending on the computer
top_path='/mnt/images/2/VOC2012';
%top_path='/mnt/images/2/bsds_segmentations/images500';
%iids = imgList(top_path, 'train');
%path=strcat(top_path,'/train');
resultpath = strcat(top_path,'/','Gradients_WHOLE_IMAGES');
mkdir(resultpath);
lmet = numel(methods);


% image test set
[iids,t]=textread(sprintf(VOCopts.seg.imgsetpath,VOCopts.trainset),'%s %d');


for i = 1:numel(iids),
    %    ids = iids(i).name(1:end-4);
    %ids = sprintf('%d', iids(i));
    ids = sprintf('%d', iids{i});
    %fids = strcat(path, '/', ids,'.jpg')
    fids = sprintf(VOCopts.imgpath,iids{i});
    im = imread(fids);
    
    
    [ih,iw,ill] = size(im);
    
    for m=1:lmet
        sprintf('%s %d', 'Algorithm id: ', m)
        try
            temp = methods{m}(im);
        catch err
            disp('algorithm error');
            temp = zeros(ih,iw);
        end
        
        %figure;
        rest = strcat(ids, '_',methods_name{m},'_result_0.bmp');
        restpath = strcat(resultpath, '/', rest);
        imwrite((temp), restpath, 'BMP');
    end
    
end
end
