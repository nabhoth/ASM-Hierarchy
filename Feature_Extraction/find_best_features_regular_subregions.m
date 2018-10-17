function [] = find_best_features_regular_subregions(type, window)
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
%methods = {'0','gc_segment', 'roi_sub_i_segment', 'salience_segment', 'pbCGTG', 'pbBGTG', 'globalPb', 'im2ucm'};
%path

%computes features for images - whole images 
%
%parameters:
%           dir_path - path to the directories containing the images 
%output:
%           mapping - array of features 

seg_path = strcat('/mnt/images/bsds_segmentations/images500/result/',type, '/regular_regions/');
img_path = strcat('/mnt/images/bsds_segmentations/images500/',type);

dids = dir(seg_path);
file_pattern=strcat(window,'x',window,'_result.mat')
for j = 1:numel(dids),

sseg_path = fullfile(seg_path, dids(j).name);
    
iids = dir(fullfile(sseg_path,'*.mat'));

segindex = 0;
for i = 1:numel(iids),
    s = regexp(iids(i).name, file_pattern);
    iids(i).name
    l = size(s);
    if l(1) > 0,
        clear mapping;
        ids = iids(i).name(1:end-4)
        load(strcat(sseg_path,'/',iids(i).name));
        ids = iids(i).name(1:end-size(file_pattern,2))
        %    im = strcat(sprintf('%d',results{1}(1)), '.jpg')
        
        
        %read the image from which to compute the features
        im = strcat(img_path,'/',ids, '.jpg')
        ims = strcat(img_path,'/',ids,'_salience_small.jpg');
        imsi = strcat(img_path,'/',ids,'_ikm_salience.jpg');
        imsg = strcat(img_path,'/',ids,'_gbvs_salience.jpg');
        ime = strcat(img_path,'/',ids,'_edge.ppm');
        
        
        %read the image from which to compute the features
        I = imread(im);
        IG = rgb2gray(I);
        IBW = im2bw(I);
        imG = im2double(I);
        imGG = im2double(IG);
        imBW = im2double(IBW);
        iiS = imread(ims);
        iiSI = imread(imsi);
        iiSG = imread(imsg);
        iiE = imread(ime);
        
        
        sprintf('processing %d regions', numel(results))
        index = 1;
        mapping = zeros(numel(results),10000);
        for reg = 1:numel(results),
            sprintf('processing %d',reg)
            mapping(index,1) = results{reg}(1);
            mapping(index,2) = results{reg}(2);
            
            mapping(index,3:6) = results{reg}(3:6);
            img = imG(results{reg}(3):results{reg}(4),results{reg}(5):results{reg}(6),:);
            imgg = imGG(results{reg}(3):results{reg}(4),results{reg}(5):results{reg}(6),:);
            iis  = iiS(results{reg}(3):results{reg}(4),results{reg}(5):results{reg}(6),:);
            iisi = iiSI(results{reg}(3):results{reg}(4),results{reg}(5):results{reg}(6),:);
            iie  = iiE(results{reg}(3):results{reg}(4),results{reg}(5):results{reg}(6),:);
            iisg = iiSG(results{reg}(3):results{reg}(4),results{reg}(5):results{reg}(6),:);
            imggg = IG(results{reg}(3):results{reg}(4),results{reg}(5):results{reg}(6),:);
            imgbw = imBW(results{reg}(3):results{reg}(4),results{reg}(5):results{reg}(6),:);
            count = 7;
            [lh, f] = his(imgg);
            mapping(index, count:count+lh(2)-1) = f;
            count = count + lh(2)
            [lf, f] = ft(imgg);
            mapping(index, count:count+lf(2)-1) = f;
            count = count + lf(2)
            [lg, f] = gab(imgg);
            mapping(index, count:count+lg(2)-1) = f;
            count = count + lg(2)
            [lw, f] = wav(imgg);
            mapping(index, count:count+lw(2)-1) = f;
            count = count + lw(2)
            [lc, f] = con(imgg);
            mapping(index, count:count+lc(2)-1) = f;
            count = count + lc(2)
            [la, f] = acu(imgg);
            mapping(index, count:count+la(2)-1) = f;
            count = count + la(2)
            [li, f] = gis(img);
            mapping(index, count:count+li(2)-1) = f;
            count = count + li(2)
            [lo, f] = ogr(img);
            mapping(index, count:count+lo(2)-1) = f;
            count = count + lo(2)
            [ls, f] = lsf(iis, iie);
            mapping(index, count:count+ls(2)-1) = f;
            count = count + ls(2)
            [ll, f] = lif(iisi, iie);
            mapping(index, count:count+ll(2)-1) = f;
            count = count + ll(2)
            [llg, f] = lgf(iisg, iie);
            mapping(index, count:count+llg(2)-1) = f;
            count = count + llg(2)
            [lb, f] = rgb(imgg, img);
            mapping(index, count:count+lb(2)-1) = f;
            count = count + lb(2)
            [lr, f] = rgf(imggg);
            mapping(index, count:count+lr(2)-1) = f;
            count = count + lr(2)
            [lrr, f] = rbf (imgbw);
            if numel(lrr) > 1,
                mapping(index, count:count+lrr(2)-1) = f;
                count = count + lrr(2)
            end;
            
            if index == 1,
                counter = count
            end
            %            mapping(index, 2:counter+1) = mapping(index, 1:counter);
            %            mapping(index,1) = str2double(ids);
            index = index + 1;
        end
        save(strcat(sseg_path,'/',ids,'_features.mat'),'mapping');
        segindex = reg+segindex-1;
    end
end
end
end

function [l, f] = his(imgg)
 sprintf('processing...')
            %sum of gray histogram
            H = imhist(imgg, 100); %min-max
            f = zeros(1,105);
            f(1,1:100) = double(H);
            %stats
            f(1,101) = std(f(1,1:100));
            f(1,102) = mean(f(1,1:100));
            f(1,103) = var(f(1,1:100));
            f(1,104) = cov(f(1,1:100));
            f(1,105) = median(f(1,1:100));
            l = size(f);
end            
            
function [l, f] = ft(imgg)           
            %sum of fft
            FH = sum(fft(imgg,100)');
            reals = real(FH);
            imags = imag(FH);
            f = zeros(1,105);
            f(1,1:100) = complex(reals,imags);
            %stats
            f(1,101) = std(f(1,1:100));
            f(1,102) = mean(f(1,1:100));
            f(1,103) = var(f(1,1:100));
            f(1,104) = cov(f(1,1:100));
            f(1,105) = median(f(1,1:100));
            l = size(f);
end

function [l, f] = gab(imgg)         
            %gabor filter
            [G,gabout] = gaborfilter1(imgg,2,4,16,4*pi/3);
            GH = imhist(im2double(gabout),100);
            f = zeros(1,105);
            f(1,1:100) = double(GH);
            %stats
            f(1,101) = std(f(1,1:100));
            f(1,102) = mean(f(1,1:100));
            f(1,103) = var(f(1,1:100));
            f(1,104) = cov(f(1,1:100));
            f(1,105) = median(f(1,1:100));
            l = size(f);
end

function [l, f]=wav(imgg)          
            %wavelet
            [cA,cH,cV,cD] = (dwt2(imgg,'haar'));
            CH = imhist(cH,100);
            CA = imhist(cA,100);
            CV = imhist(cV,100);
            CD = imhist(cD,100);
            f = zeros(1,420);
            f(1,1:100) = CH;
            %stats
            f(1,101) = std(f(1,1:100));
            f(1,102) = mean(f(1,1:100));
            f(1,103) = var(f(1,1:100));
            f(1,104) = cov(f(1,1:100));
            f(1,105) = median(f(1,1:100));
            
            f(1,106:205) = CA;
            %stats
            f(1,206) = std(f(1,106:205));
            f(1,207) = mean(f(1,106:205));
            f(1,208) = var(f(1,106:205));
            f(1,209) = cov(f(1,106:205));
            f(1,210) = median(f(1,106:205));
            
            f(1,211:310) = CV;
            %stats
            f(1,311) = std(f(1,211:310));
            f(1,312) = mean(f(1,211:310));
            f(1,313) = var(f(1,211:310));
            f(1,314) = cov(f(1,211:310));
            f(1,315) = median(f(1,211:310));
            
            f(1,316:415) = CD;
            %stats
            f(1,416) = std(f(1,316:415));
            f(1,417) = mean(f(1,316:415));
            f(1,418) = var(f(1,316:415));
            f(1,419) = cov(f(1,316:415));
            f(1,420) = median(f(1,316:415));
            l = size(f);
end

function [l, f] = con(imgg)
            %contrast
            f = contrast(imgg)/10^4;
            l = size(f);
end

function [l, f] = acu(imgg)
            %acutance
            f = acutance(imgg)*10^4;
            l = size(f);
end
function [l, f] = gis(img) 
            %Gist
            [h,w,l] = size(img);
            A = min(h,w);
            h = gist_features(img);%,min(h,w));
            lh = size(h);
            f = zeros(1, lh(2)+5);
            f(1,1:lh(1)) = h;
            %stats
            f(1,lh(2)+1) = std(h);
            f(1,lh(2)+2) = mean(h);
            f(1,lh(2)+3) = var(h);
            f(1,lh(2)+4) = cov(h);
            f(1,lh(2)+5) = median(h);            
            l = size(f);
end

function [l, f] = ogr(img)
            %Orient Gradient
            [BGa,BGb] = detBG(img);
            [TGa,TGb] = detTG(img);
            [CGa,pbhist] = detCG(img);
            f = zeros(1,415);
            cou = 1;
            for dr = 1:8,
                BGA = imhist(BGa(:,:,dr),128);
                f(1,cou:cou+127) = BGA;
                cou = cou + 128;
                %stats
                f(1,cou) = std(BGA);
                f(1,cou+1) = mean(BGA);
                f(1,cou+2) = var(BGA);
                f(1,cou+3) = cov(BGA);
                f(1,cou+4) = median(BGA);
                cou = cou + 5;
                
                TGA = imhist(TGa(:,:,dr),128);
                f(1,cou:cou+127) = TGA;
                cou = cou + 128;
                %stats
                f(1,cou) = std(TGA);
                f(1,cou+1) = mean(TGA);
                f(1,cou+2) = var(TGA);
                f(1,cou+3) = cov(TGA);
                f(1,cou+4) = median(TGA);
                cou = cou + 5;
                
                for color = 1:3,
                    COLRS = imhist(CGa(:,:,color,dr),128);
                    f(1,cou:cou+127) = COLRS;
                    cou = cou + 128;
                    %stats
                    f(1,cou) = std(COLRS);
                    f(1,cou+1) = mean(COLRS);
                    f(1,cou+2) = var(COLRS);
                    f(1,cou+3) = cov(COLRS);
                    f(1,cou+4) = median(COLRS);
                    cou = cou + 5;
                end
            end            
            l = size(f);
end
function [l, f] = lsf(iis, iie)            
            %line feature 2069-2188 from Saliency 1
            f = zeros(1,133);
            [m,n,l] = size(iis);
            %m = abs(mapping(i,3)-mapping(i,4));
            %n = abs(mapping(i,5)-mapping(i,6));
            offset = 0;
            linecou = 0;
            %line feature
           
            for linei = 17:17:80
                for linej = 3:--2:-1
                    for lineK = 0:20:40
                        if(lineK == 0)
                            linek = 10;
                        else
                            linek = lineK;
                        end
                        
                        %i,j,k = thresh,radius,length
                        [lines,pixels] = longlines(iis,iie,linei,linej,linek);
                        
                        binarypix = pixels/128;
                        Lpx = sum(sum(binarypix));
                        
                        if(Lpx == 0)
                            f(1, offset+linecou : offset+5+linecou) = 0;
                        else
                            %linepixels saliency nomalize
                            f(1, offset   + linecou) = sum(sum(lines(:,:,3)))/Lpx;
                            %sum linepixels
                            f(1, offset+1 + linecou) = Lpx;
                            %linepixels/allpixels
                            f(1, offset+2 + linecou) = Lpx/(n*m);
                            %linepixels(i,j,k)/linepixels(20,3,10)
                            f(1, offset+3 + linecou) = Lpx/f(1, offset + 1);
                            %different
                            dif = line_distinct(lines,iis,1,m,1,n);
                            f(1, offset+4 + linecou) = dif;
                        end
                        linecou = linecou + 5;
                    end
                end
            end
            %stats
            f(1,128) = std(f(1,1:120));
            f(1,129) = mean(f(1,1:120));
            f(1,130) = var(f(1,1:120));
            f(1,131) = cov(f(1,1:120));
            f(1,132) = median(f(1,1:120));
            l = size(f);
end

function [l, f] = lif(iis, iie)            
             %line feature 2189-2307 from Saliency Itty
            [m,n,l] = size(iis);
            f = zeros(1,125);
            %n = abs(mapping(i,5)-mapping(i,6));
            offset = 0;
            linecou = 0;
            for linei = 17:17:80
                for linej = 3:--2:-1
                    for lineK = -3:17:40
                        if(lineK == 0)
                            linek = 10;
                        else
                            linek = lineK;
                        end
                        
                        %i,j,k = thresh,radius,length
                        [lines,pixels] = longlines(iisi,iie,linei,linej,linek);
                        
                        binarypix = pixels/128;
                        Lpx = sum(sum(binarypix));
                        
                        if(Lpx == 0)
                            f(1, offset+linecou : offset+5+linecou) = 0;
                        else
                            %linepixels saliency nomalize
                            f(1, offset   + linecou) = sum(sum(lines(:,:,3)))/Lpx;
                            %sum linepixels
                            f(1, offset+1 + linecou) = Lpx;
                            %linepixels/allpixels
                            f(1, offset+2 + linecou) = Lpx/(n*m);
                            %linepixels(i,j,k)/linepixels(20,3,10)
                            f(1, offset+3 + linecou) = Lpx/f(1, offset + 1);
                            %different
                            dif = line_distinct(lines,iis,1,m,1,n);
                            f(1, offset+4 + linecou) = dif;
                        end
                        linecou = linecou + 5;
                    end
                end
            end
            %stats
            f(1,121) = std(f(1,1:120));
            f(1,122) = mean(f(1,1:120));
            f(1,123) = var(f(1,1:120));
            f(1,124) = cov(f(1,1:120));
            f(1,125) = median(f(1,1:120));            
            l = size(f);
end

function [l, f] = lgf(iis, iie)          
            %line feature 2308-2427 from Saliency GBVS
            [m,n,l] = size(iis);
            f = zeros(1,125);
            %m = abs(mapping(i,3)-mapping(i,4));
            %n = abs(mapping(i,5)-mapping(i,6));
            offset = 0;
            linecou = 0;
            for linei = 17:17:80
                for linej = 3:--2:-1
                    for lineK = -3:17:40
                        if(lineK == 0)
                            linek = 10;
                        else
                            linek = lineK;
                        end
                        
                        %i,j,k = thresh,radius,length
                        [lines,pixels] = longlines(iisg,iie,linei,linej,linek);
                        
                        binarypix = pixels/128;
                        Lpx = sum(sum(binarypix));
                        
                        if(Lpx == 0)
                            f(1, offset+linecou : offset+5+linecou) = 0;
                        else
                            %linepixels saliency nomalize
                            f(1, offset   + linecou) = sum(sum(lines(:,:,3)))/Lpx;
                            %sum linepixels
                            f(1, offset+1 + linecou) = Lpx;
                            %linepixels/allpixels
                            f(1, offset+2 + linecou) = Lpx/(n*m);
                            %linepixels(i,j,k)/linepixels(20,3,10)
                            f(1, offset+3 + linecou) = Lpx/f(1, offset + 1);
                            %different
                            dif = line_distinct(lines,iis,1,m,1,n);
                            f(counter, offset+4 + linecou) = dif;
                        end
                        linecou = linecou + 5;
                    end
                end
            end
            %stats
            f(1,121) = std(f(1,1:120));
            f(1,122) = mean(f(1,1:120));
            f(1,123) = var(f(1,1:120));
            f(1,124) = cov(f(1,1:120));
            f(1,125) = median(f(1,1:120));
            l = size(f);
end

function [l, f] = rgb(imgg, img)
            %RGBhistogram  2427-2626
            f = zeros(1,410);
            h = imhist(imgg, 100);
            f(1,1:100) = h;
            %stats
            f(1,101) = std(f(1,1:100));
            f(1,102) = mean(f(1,1:100));
            f(1,103) = var(f(1,1:100));
            f(1,104) = cov(f(1,1:100));
            f(1,105) = median(f(1,1:100));
            
            h = imhist(img(:,:,1), 100);
            f(1,106:205) = h;
            %stats
            f(1,206) = std(f(1,106:205));
            f(1,207) = mean(f(1,106:205));
            f(1,208) = var(f(1,106:205));
            f(1,209) = cov(f(1,106:205));
            f(1,210) = median(f(1,106:205));
            
            h = imhist(img(:,:,2), 100);
            f(1,211:310) = h;
            %stats
            f(1,311) = std(f(1,211:310));
            f(1,312) = mean(f(1,211:310));
            f(1,313) = var(f(1,211:310));
            f(1,314) = cov(f(1,211:310));
            f(1,315) = median(f(1,211:310));
            
            h = imhist(img(:,:,3), 100);
            f(1,316:415) = h;
            %stats
            f(1,416) = std(f(1,316:415));
            f(1,417) = mean(f(1,316:415));
            f(1,418) = var(f(1,316:415));
            f(1,419) = cov(f(1,316:415));
            f(1,420) = median(f(1,316:415));
            l = size(f);
end

function [l, f] = rgf(imggg)
            %region properties
            [h,w,l] = size(imggg)
            diagonale = numel(diag(imggg));
            regprops = regionprops(imggg,'all');
            elements = numel(regprops);
            %number of regions from BW level image
            f = zeros(1,605);
            f(1,1) = double(elements)/(h*w);
            %additional region statistics
            %number of regions
            %mapping(i,data) = elements;
            area = zeros(1, elements);
            cw = zeros(1, elements);
            ch = zeros(1, elements);
            bb = zeros(1, elements);
            mal = zeros(1, elements);
            mil = zeros(1, elements);
            ecc = zeros(1, elements);
            o = zeros(1, elements);
            cva = zeros(1, elements);
            sol = zeros(1, elements);
            ext = zeros(1, elements);
            for u=1:elements,
                area(u) = double(regprops(u).Area)/(h*w);
                cw(u) = double(regprops(u).Centroid(1))/w;
                ch(u) = double(regprops(u).Centroid(2))/h;
                bb(u) = double(regprops(u).BoundingBox(3)*regprops(u).BoundingBox(4))/(h*w);
                mal(u) = double(regprops(u).MajorAxisLength)/diagonale;
                mil(u) = double(regprops(u).MinorAxisLength)/diagonale;
                ecc(u) = regprops(u).Eccentricity;
                o(u) = double(regprops(u).Orientation)/360;
                cva(u) = double(regprops(u).ConvexArea)/(h*w);
                sol(u) = regprops(u).Solidity;
                ext(u) = regprops(u).Extent;
            end
            h = hist(area, 50);
            f(1,2:51) = h;
            %stats
            f(1,52) = std(f(1,2:51));
            f(1,53) = mean(f(1,2:51));
            f(1,54) = var(f(1,2:51));
            f(1,55) = cov(f(1,2:51));
            f(1,56) = median(f(1,2:51));           
            h = hist(cw, 50);
            f(1,57:106) = h;
            %stats
            f(1,107) = std(f(1,57:106));
            f(1,108) = mean(f(1,57:106));
            f(1,109) = var(f(1,57:106));
            f(1,110) = cov(f(1,57:106));
            f(1,111) = median(f(1,57:106));
            h = hist(ch, 50);
            f(1,112:161) = h;
            %stats
            f(1,162) = std(f(1,112:161));
            f(1,163) = mean(f(1,112:161));
            f(1,164) = var(f(1,112:161));
            f(1,165) = cov(f(1,112:161));
            f(1,166) = median(f(1,112:161));
            h = hist(bb, 50);
            f(1,166:215) = h;
            %stats
            f(1,216) = std(f(1,166:215));
            f(1,217) = mean(f(1,166:215));
            f(1,218) = var(f(1,166:215));
            f(1,219) = cov(f(1,166:215));
            f(1,220) = median(f(1,166:215));
            h = hist(mal, 50);
            f(1,221:270) = h;
            %stats
            f(1,271) = std(f(1,221:270));
            f(1,272) = mean(f(1,221:270));
            f(1,273) = var(f(1,221:270));
            f(1,274) = cov(f(1,221:270));
            f(1,275) = median(f(1,221:270));
            h = hist(mil, 50);
            f(1,276:325) = h;
            %stats
            f(1,326) = std(f(1,276:325));
            f(1,327) = mean(f(1,276:325));
            f(1,328) = var(f(1,276:325));
            f(1,329) = cov(f(1,276:325));
            f(1,330) = median(f(1,276:325));
            h = hist(ecc, 50);
            f(1,331:380) = h;
            %stats
            f(1,381) = std(f(1,331:380));
            f(1,382) = mean(f(1,331:380));
            f(1,383) = var(f(1,331:380));
            f(1,384) = cov(f(1,331:380));
            f(1,385) = median(f(1,331:380));
            h = hist(o, 50);
            f(1,386:435) = h;
            %stats
            f(1,436) = std(f(1,386:435));
            f(1,437) = mean(f(1,386:435));
            f(1,438) = var(f(1,386:435));
            f(1,439) = cov(f(1,386:435));
            f(1,440) = median(f(1,386:435));
            h = hist(cva, 50);
            f(1,441:490) = h;
            %stats
            f(1,491) = std(f(1,441:490));
            f(1,492) = mean(f(1,441:490));
            f(1,493) = var(f(1,441:490));
            f(1,494) = cov(f(1,441:490));
            f(1,495) = median(f(1,441:490));
            h = hist(sol, 50);
            f(1,496:545) = h;
            %stats
            f(1,546) = std(f(1,496:545));
            f(1,547) = mean(f(1,496:545));
            f(1,548) = var(f(1,496:545));
            f(1,549) = cov(f(1,496:545));
            f(1,550) = median(f(1,496:545));
            h = hist(ext, 50);
            f(1,551:600) = h;
            %stats
            f(1,601) = std(f(1,551:600));
            f(1,602) = mean(f(1,551:600));
            f(1,603) = var(f(1,551:600));
            f(1,604) = cov(f(1,551:600));
            f(1,605) = median(f(1,551:600));
            l = size(f);
end

function [l, f] = rbf (imgbw)
            %region properties
            [h,w,l] = size(imgbw)
            f = zeros(1,605);
            diagonale = numel(diag(imgbw));
            regprops = regionprops(imgbw,'all');
            elements = numel(regprops);
            %number of regions from grey level image
            f(1,1) = double(elements)/(h*w);
            %additional region statistics
            if elements > 0,
            for u=1:elements,
                are(u) = double(regprops(u).Area)/(h*w);
                cw(u) = double(regprops(u).Centroid(1))/w;
                ch(u) = double(regprops(u).Centroid(2))/h;
                bb(u) = double(regprops(u).BoundingBox(3)*regprops(u).BoundingBox(4))/(h*w);
                mal(u) = double(regprops(u).MajorAxisLength)/diagonale;
                mil(u) = double(regprops(u).MinorAxisLength)/diagonale;
                ecc(u) = regprops(u).Eccentricity;
                o(u) = double(regprops(u).Orientation)/360;
                cva(u) = double(regprops(u).ConvexArea)/(h*w);
                sol(u) = regprops(u).Solidity;
                ext(u) = regprops(u).Extent;
            end
            h = hist(are, 50);
            f(1,2:51) = h;
            %stats
            f(1,52) = std(f(1,2:51));
            f(1,53) = mean(f(1,2:51));
            f(1,54) = var(f(1,2:51));
            f(1,55) = cov(f(1,2:51));
            f(1,56) = median(f(1,2:51));           
            h = hist(cw, 50);
            f(1,57:106) = h;
            %stats
            f(1,107) = std(f(1,57:106));
            f(1,108) = mean(f(1,57:106));
            f(1,109) = var(f(1,57:106));
            f(1,110) = cov(f(1,57:106));
            f(1,111) = median(f(1,57:106));
            h = hist(ch, 50);
            f(1,112:161) = h;
            %stats
            f(1,162) = std(f(1,112:161));
            f(1,163) = mean(f(1,112:161));
            f(1,164) = var(f(1,112:161));
            f(1,165) = cov(f(1,112:161));
            f(1,166) = median(f(1,112:161));
            h = hist(bb, 50);
            f(1,166:215) = h;
            %stats
            f(1,216) = std(f(1,166:215));
            f(1,217) = mean(f(1,166:215));
            f(1,218) = var(f(1,166:215));
            f(1,219) = cov(f(1,166:215));
            f(1,220) = median(f(1,166:215));
            h = hist(mal, 50);
            f(1,221:270) = h;
            %stats
            f(1,271) = std(f(1,221:270));
            f(1,272) = mean(f(1,221:270));
            f(1,273) = var(f(1,221:270));
            f(1,274) = cov(f(1,221:270));
            f(1,275) = median(f(1,221:270));
            h = hist(mil, 50);
            f(1,276:325) = h;
            %stats
            f(1,326) = std(f(1,276:325));
            f(1,327) = mean(f(1,276:325));
            f(1,328) = var(f(1,276:325));
            f(1,329) = cov(f(1,276:325));
            f(1,330) = median(f(1,276:325));
            h = hist(ecc, 50);
            f(1,331:380) = h;
            %stats
            f(1,381) = std(f(1,331:380));
            f(1,382) = mean(f(1,331:380));
            f(1,383) = var(f(1,331:380));
            f(1,384) = cov(f(1,331:380));
            f(1,385) = median(f(1,331:380));
            h = hist(o, 50);
            f(1,386:435) = h;
            %stats
            f(1,436) = std(f(1,386:435));
            f(1,437) = mean(f(1,386:435));
            f(1,438) = var(f(1,386:435));
            f(1,439) = cov(f(1,386:435));
            f(1,440) = median(f(1,386:435));
            h = hist(cva, 50);
            f(1,441:490) = h;
            %stats
            f(1,491) = std(f(1,441:490));
            f(1,492) = mean(f(1,441:490));
            f(1,493) = var(f(1,441:490));
            f(1,494) = cov(f(1,441:490));
            f(1,495) = median(f(1,441:490));
            h = hist(sol, 50);
            f(1,496:545) = h;
            %stats
            f(1,546) = std(f(1,496:545));
            f(1,547) = mean(f(1,496:545));
            f(1,548) = var(f(1,496:545));
            f(1,549) = cov(f(1,496:545));
            f(1,550) = median(f(1,496:545));
            h = hist(ext, 50);
            f(1,551:600) = h;
            %stats
            f(1,601) = std(f(1,551:600));
            f(1,602) = mean(f(1,551:600));
            f(1,603) = var(f(1,551:600));
            f(1,604) = cov(f(1,551:600));
            f(1,605) = median(f(1,551:600));
            l= size(f);
            end
end

