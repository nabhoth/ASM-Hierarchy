path = 'D:\/bsds_segmentations/images/test/results/color/result';

iids = imgList('test');
max_regions = numel(iids)*20;
mapping = zeros(max_regions,127);

segindex = 0;
for ix = 1:numel(iids),
ids = iids(ix)
im = strcat(sprintf('%d',ids), '.jpg');
im_data = strcat(sprintf('%d',ids), '.mat');
ims = strcat(sprintf('%d',ids),'_salience_small.jpg');
ime = strcat(sprintf('%d',ids),'_edge.jpg');

load(sprintf('%s/%s', path, im_data), 'results')

%read the image from which to compute the features
I = imread(im);
IG = rgb2gray(I);
IMS = imread(ims);

imG = im2double(I);
imGG = im2double(IG);


%ixx = ix - 1;
for reg = 1:numel(results), 
    index = reg+segindex;
    try
        %image id
        mapping(index,1) = results{reg}(1);
        %algorithm id
        mapping(index,2) = results{reg}(2);
        
    catch err
        break;
    end
    %image region
    mapping(index,3:6) = results{reg}(3:6);
  
    img = imG(results{reg}(3):results{reg}(4),results{reg}(5):results{reg}(6),:);
    imgg = imGG(results{reg}(3):results{reg}(4),results{reg}(5):results{reg}(6),:);

    
    [h,w,l] = size(img);
    IMS_size = size(IMS);
    m = abs(mapping(index,3)-mapping(index,4));
    n = abs(mapping(index,5)-mapping(index,6));
    mapping(index,7) = n*m;
    cou = 0;

    %line feature
    for(i = 20:20:80)   
        for(j = 3:-1:2)              
            for(K = 0:20:40) 
                if(K == 0)
                    k = 10;
                else
                    k = K;
                end
                
                 %i,j,k = thresh,radius,length
                [lines,pixels] = longlines(ims,ime,i,j,k,results{reg}(3),results{reg}(4),results{reg}(5),results{reg}(6));
                
                %linepixels has value 128 so divide 128
                binarypix = pixels/128;
                Lpx = sum(sum(binarypix));
                                   
                
                if(Lpx == 0)
                    mapping(index, 8+cou : 12+cou) = 0;
                else
                    %linepixels saliency nomalize  
                    mapping(index, 8 + cou) = sum(sum(lines(:,:,3)))/Lpx;                             
                    %sum linepixels 
                    mapping(index, 9 + cou) = Lpx;
                    %linepixels/allpixels
                    mapping(index, 10 + cou) = Lpx/(n*m) * 100;              
                    %linepixels(i,j,k)/linepixels(20,3,10)
                    mapping(index, 11 + cou) = mapping(index, 9 + cou)/mapping(index, 9) * 100;
                    %line_distinguish
                    [dif] = line_distinct(lines,IMS,results{reg}(3),results{reg}(4),results{reg}(5),results{reg}(6));
                    mapping(index, 12 + cou) = dif;
                end
                
                cou = cou + 5;              
            end           
        end
    end
end
    segindex = reg+segindex-1
    save('D:\/image_transforms/test/mapping_linefeature_region', 'mapping');
end
save('D:\/image_transforms/test/mapping_linefeature_region', 'mapping');