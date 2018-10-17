function [ mapping] = find_best_features_subregions(set)
%splits image according to the xpoints and ypoints
%for each generates a sub image and apply the segmentations from the
%possible methods and keeps the best
%this function uses pieces of code (for evaluation of segmentation)
%from the bsds databenchmark suite programed by Martin
%
%parameters:
%           imag - the name of the image (5-10) is usual value
%           divider - indicates the density of the image splitting
%           methods - array of functions
%output:
%           result - array of probabilities with segments coordinates, algorithms id, and scores and
%           image id
%methods = {'gc_segment', 'ncut_segment', 'roi_sub_i_segment', 'pbCGTG', 'pbBGTG', 'im2ucm'};
%methods = {'gc_segment', 'ncut_segment', 'roi_sub_i_segment', 'salience_segment', 'pbCGTG', 'pbBGTG', 'globalPb', 'im2ucm'};
%path
if strcmp(set, 'train'),
    path='/mnt/hd1/bsds_segmentations/images/train/results';
    iids = imgList('train');
else
    path='/mnt/hd1/bsds_segmentations/images/test/results/color/result';
    iids = imgList('test');
end
%mkdir (path, 'results')
%[ID, Images] = textread('/home/nabhoth/codes/bsds_segmentations/New_Image_Order_Color_min.txt','%f %f','headerlines',1);
%O = [ID,Images];
%O = sortrows(O,1)

max_regions = numel(iids)*20;
mapping = zeros(max_regions,1669);

segindex = 0;
for ix = 1:numel(iids),
    ids = iids(ix)
    im = strcat(sprintf('%d',ids), '.jpg');
    try
        
        im_data = strcat(sprintf('%d',ids), '_result.mat');
        
        load(sprintf('%s/%s', path, im_data), 'results')
        
        %read the image from which to compute the features
        I = imread(im);
        IG = rgb2gray(I);
        imG = im2double(I);
        imGG = im2double(IG);
        
        %ixx = ix - 1;
        for reg = 1:numel(results),
            index = reg+segindex;
 %           try
                %image id
                mapping(index,1) = results{reg}(1);
                %algorithm id
                mapping(index,2) = results{reg}(2);
                
 %           catch err
 %               break;
 %           end
            %image region
            mapping(index,3:6) = results{reg}(3:6);
            img = imG(results{reg}(3):results{reg}(4),results{reg}(5):results{reg}(6),:);
            ig = I(results{reg}(3):results{reg}(4),results{reg}(5):results{reg}(6),:);
            imgg = imGG(results{reg}(3):results{reg}(4),results{reg}(5):results{reg}(6),:);
            igg = IG(results{reg}(3):results{reg}(4),results{reg}(5):results{reg}(6),:);
            
            [h,w,l] = size(img);
            %sum of histogram
            H = sum(hist(img, 100)');
            mapping(index,7:106) = H(:);
            %sum of fft
            FH = sum(fft(imgg,100)');
            mapping(index,107:206) = FH;
            
            %gabor filter
            [G,gabout] = gaborfilter1(imgg,2,4,16,4*pi/3);
            histo = hist(im2double(gabout),100);
            GH = sum(h');
            mapping(index,207:306) = GH;
            
            %haar filter
            [cA,cH,cV,cD] = (dwt2(imgg,'haar'));
            CH = sum(hist(cH,100)');
            CA = sum(hist(cA,100)');
            CV = sum(hist(cV,100)');
            CD = sum(hist(cD,100)');
            mapping(index,307:406) = CH;
            mapping(index,407:506) = CA;
            mapping(index,507:606) = CV;
            mapping(index,607:706) = CD;
            %wavelets
            
            %contrast
            
            mapping(index,707) = contrast(imgg);
            
            
            %salience mean and entropy
            %im_sal = imread(strcat(sprintf('%d',ids), '_salience_small.jpg'));
            
            %acutance
            mapping(index,708) = acutance(imgg);
            
            %gist
            g = gist_features(img)';
            mapping(index,709:1668) = g(:);
            
        end
        segindex = reg+segindex-1;
    catch error
    end
end