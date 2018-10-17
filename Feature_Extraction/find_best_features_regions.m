function [ mapping] = find_best_features_regions(dir_path)
%computes features for images - whole images 
%
%parameters:
%           dir_path - path to the directories containing the images 
%output:
%           mapping - array of features 

file_pattern='\d\d\d\d_\d\d\d\d\d\d.jpg';
iids = dir(fullfile(dir_path, '*jpg'));
mapping = zeros(numel(iids),6000);
counter = 1;

for i = 1:numel(iids),
    s = regexp(iids(i).name, file_pattern);
    l = size(s)
    if l(1) > 0,
        ids = iids(i).name(1:end-4);
        
        mapping(counter,1) = ids;
        %read the various images
        if isnumeric(ids),
            im = strcat(path,'/',sprintf('%d',ids), '.jpg');
            ims = strcat(path,'/',sprintf('%d',ids),'_salience_small.jpg');
            imsi = strcat(path,'/',sprintf('%d',ids),'_ikm_salience.jpg');
            imsg = strcat(path,'/',sprintf('%d',ids),'_gbvs_salience.jpg');
            ime = strcat(path,'/',sprintf('%d',ids),'_edge.ppm');
        else
            im = strcat(path,'/',ids, '.jpg');
            ims = strcat(path,'/',ids,'_salience_small.jpg');
            imsi = strcat(path,'/',ids,'_ikm_salience.jpg');
            imsg = strcat(path,'/',ids,'_gbvs_salience.jpg');
            ime = strcat(path,'/',ids,'_edge.ppm');
        end
        
        
        %read the image from which to compute the features
        I = imread(im);
        IG = rgb2gray(I);
        IBW = im2bw(I);
        imG = im2double(I);
        imGG = im2double(IG);
        imBW = im2double(IBW);
        imgbw = imBW;
        iis = imread(ims);
        iie = imread(ime);
        
        img = im2double(I);
        imgg = im2double(IG);
        imggg = IG;
        [h,w] = size(IG);
        % %sum of histogram
        % H = sum(hist(img, 100)');
        % mapping(ix,3:102) = H(:);
        % %sum of fft
        % FH = sum(fft(imgg,100)');
        % mapping(ix,103:202) = FH;
        %
        % %gabor filter
        % [G,gabout] = gaborfilter1(imgg,2,4,16,4*pi/3);
        % histo = hist(im2double(gabout),100);
        % GH = sum(histo');
        % mapping(ix,203:302) = GH;
        %
        % %haar filter
        % [cA,cH,cV,cD] = (dwt2(imgg,'haar'));
        % CH = sum(hist(cH,100)');
        % CA = sum(hist(cA,100)');
        % CV = sum(hist(cV,100)');
        % CD = sum(hist(cD,100)');
        % mapping(ix,303:402) = CH;
        % mapping(ix,403:502) = CA;
        % mapping(ix,503:602) = CV;
        % mapping(ix,603:702) = CD;
        % %wavelets
        %
        % %contrast
        % %histogram of the gray scale
        % histo = hist(imgg, 256);
        % %do cumulative histogram
        % %first sum values for all columns
        % for i = 2:w,
        %     histo(:,i) = histo(:,i)+histo(:,i-1);
        % end
        % h1 = histo(:,w);
        % %cum ulatively sum all rows(bins)
        % for i = 2:256,
        %     h1(i,:) = h1(i,:)+h1(i-1,:);
        % end
        % h2 = h1;
        % %h =
        % X = zeros(2,256);
        % for i = 1:256,
        %     X(1,i) = 1; X(2,i) = i;
        % end;
        % tX = X';
        % a1 = 0;a2 = 0;a3 = 0;a4 = 0;
        % for i = 1:256,
        %     a1 = a1 + (tX(i,1)*X(1,i));a2 = a2 + (tX(i,1)*X(2,i));a3 = a3 + (tX(i,2)*X(1,i));a4 = a4 + (tX(i,2)*X(2,i));
        % end;
        %
        % b1 = 0;b2 = 0;
        % for i = 1:256,
        %     b1 = b1 + tX(i,1)*h2(i);
        %     b2 = b2 + tX(i,2)*h2(i);
        % end;
        %
        % c1 = (a1*b2-a3*b1)/(a1*a4-a2*a3);
        % c0 = (b1-a2*(c1))/a1;
        %
        % rms = 0.0;
        % for i = 1:256,
        %     rms = rms + (h2(i)-(c0-c1*i))^2;
        % end;
        %
        % rms = rms/256;
        % rms = sqrt(rms);
        % mapping(ix,703) = rms;
        %
        %
        % %salience mean and entropy
        % %im_sal = imread(strcat(sprintf('%d',ids), '_salience_small.jpg'));
        %
        % %acutance
        % gdiff = 0.0;
        % %average image
        % iav = zeros(h,w);
        % for i = 2:h
        %     iav(h,1) = xor(iav(h-1,1),1);
        % end
        % for j = 2:w
        %     iav(:,j) = xor(iav(:,j-1),1);
        % end
        % iav = iav*255;
        % for i = 2:h-1
        %     diff = 0;
        %     for j = 2:w-1
        %     diff = diff + abs(iav(i,j) - iav(i,j-1));
        %     diff = diff + abs(iav(i,j) - iav(i,j+1));
        %     diff = diff + abs(iav(i,j) - iav(i-1,j-1));
        %     diff = diff + abs(iav(i,j) - iav(i-1,j+1));
        %     diff = diff + abs(iav(i,j) - iav(i+1,j-1));
        %     diff = diff + abs(iav(i,j) - iav(i+1,j+1));
        %     diff = diff + abs(iav(i,j) - iav(i-1,j));
        %     diff = diff + abs(iav(i,j) - iav(i+1,j));
        %     end;
        %     gdiff = gdiff + diff^2;
        % end;
        % %size(iav_diff)
        % %size(gdiff)
        % %gdiff
        % iav_diff = gdiff/((h-2)*(w-2)*8);
        %
        % gdiff = 0.0;
        % for i = 2:h-1
        %     diff = 0;
        %     for j = 2:w-1
        %     diff = diff + abs(imgg(i,j) - imgg(i,j-1));
        %     diff = diff + abs(imgg(i,j) - imgg(i,j+1));
        %     diff = diff + abs(imgg(i,j) - imgg(i-1,j-1));
        %     diff = diff + abs(imgg(i,j) - imgg(i-1,j+1));
        %     diff = diff + abs(imgg(i,j) - imgg(i+1,j-1));
        %     diff = diff + abs(imgg(i,j) - imgg(i+1,j+1));
        %     diff = diff + abs(imgg(i,j) - imgg(i-1,j));
        %     diff = diff + abs(imgg(i,j) - imgg(i+1,j));
        %     end;
        %     gdiff = gdiff + diff^2;
        % end;
        % gdiff = 10^4*gdiff/((h-2)*(w-2)*8*2*iav_diff);
        %
        % mapping(ix,704) = gdiff;
        %
        % g = gist_features(img)';
        % mapping(index,705:1665) = g;
        
        
        [h,w,l] = size(img);
        if w == 1||h == 1,
            mapping(i,7) = -1;
        else
            sprintf('processing...')
            %sum of gray histogram 7-106
            H = sum(hist(imgg, 100)'); %min-max
            %normalize
            s = size(H);
            su = sum(H(:));%/s(2);
            h = double(H)./su;
            mapping(i,7:106) = h;
            
            
            
            %sum of fft 107-206
            FH = sum(fft(imgg,100)');
            mags = sum(abs(FH));%/100;
            reals = real(FH)/mags;
            imags = imag(FH)/mags;
            FH = complex(reals,imags);
            mapping(i,107:206) = FH;
            
            %gabor filter 207-306
            [G,gabout] = gaborfilter1(imgg,2,4,16,4*pi/3);
            histo = hist(im2double(gabout),100);
            GH = sum(histo');
            s = size(GH);
            su = sum(GH(:));%/s(2);
            gh = double(GH)./su;
            mapping(i,207:306) = gh;
            
            %wavelet 307-706
            [cA,cH,cV,cD] = (dwt2(imgg,'haar'));
            CH = sum(hist(cH,100)');
            CA = sum(hist(cA,100)');
            CV = sum(hist(cV,100)');
            CD = sum(hist(cD,100)');
            
            s = size(CH);
            su = sum(CH(:));%/s(2);
            ch = double(CH)./su;
            mapping(i,307:406) = ch;
            
            s = size(CA);
            su = sum(CA(:));%/s(2);
            ca = double(CA)./su;
            mapping(i,407:506) = ca;
            
            s = size(CV);
            su = sum(CV(:));%/s(2);
            cv = double(CV)./su;
            mapping(i,507:606) = cv;
            
            s = size(CD);
            su = sum(CD(:));%/s(2);
            cd = double(CD)./su;
            mapping(i,607:706) = cd;
            
            %contrast 707
            conts = contrast(imgg)/10^4;
            mapping(i,707) = conts;
            
            %acutance 708
            acuts = acutance(imgg)*10^4;
            mapping(i,708) = acuts;
            
            
            
            %Gist 709-1668
            [h,w,l] = size(img);
            A = min(h,w);
            mapping(i,709:1668) = gist_features(img);%,min(h,w));
            
            %Orient Gradient 1669:2068
            [BGa,BGb] = detBG(img);
            [TGa,TGb] = detTG(img);
            [CGa,pbhist] = detCG(img);
            %[CGa,pbhist] = pbCG(img);
            cou = 1668;
            scou = 3602;
            for dir = 1:8,
                BGA = sum(hist(BGa(:,:,dir),10)');
                s = size(BGA);
                su = sum(BGA(:));%/s(2);
                bga = double(BGA)./su;
                mapping(i,cou+1 :cou+10) = bga;
                cou = cou + 10;
                TGA = sum(hist(TGa(:,:,dir),10)');
                s = size(TGA);
                su = sum(TGA(:));%/s(2);
                tga = double(TGA)./su;
                mapping(i,cou+1:cou+10) = tga;
                cou = cou + 10;
                for color = 1:3,
                    COLRS = sum(hist(CGa(:,:,color,dir),10)');
                    s = size(COLRS);
                    su = sum(COLRS(:));%/s(2);
                    colrs = double(COLRS)./su;
                    mapping(i,cou+1:cou+10) = colrs;
                    
                    mapping(i,scou) = std(colrs);
                    mapping(i,scou+1) = mean(colrs);
                    mapping(i,scou+2) = var(colrs);
                    mapping(i,scou+3) = cov(colrs);
                    mapping(i,scou+4) = median(colrs);
                    scou = scou + 5;
                    cou = cou + 10;
                    %mapping(i,cou+81+color*80:cou+90+color*80) = colrs;
                end
            end
            %scou = 3602+8*15 = 3602+120 = 3722;
            
            
            %line feature 2069-2188
            [m,n,l] = size(iis);
            %m = abs(mapping(i,3)-mapping(i,4));
            %n = abs(mapping(i,5)-mapping(i,6));
            
            offset = 2069;
            linecou = 0;
            for linei = 20:20:80
                for linej = 3:-1:2
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
                            mapping(i, offset+linecou : offset+5+linecou) = 0;
                        else
                            %linepixels saliency nomalize
                            mapping(i, offset   + linecou) = sum(sum(lines(:,:,3)))/Lpx;
                            %sum linepixels
                            mapping(i, offset+1 + linecou) = Lpx;
                            %linepixels/allpixels
                            mapping(i, offset+2 + linecou) = Lpx/(n*m);
                            %linepixels(i,j,k)/linepixels(20,3,10)
                            mapping(i, offset+3 + linecou) = Lpx/mapping(i, offset + 1);
                            %different
                            dif = line_distinct(lines,iis,1,m,1,n);
                            mapping(i, offset+4 + linecou) = dif;
                        end
                        linecou = linecou + 5;
                    end
                end
            end
            
            s = 2188-2069+1;
            su = sum(mapping(i,2069:2188));%/s;
            lins = double(mapping(i,2069:2188))./su;
            mapping(i,2069:2188) = lins;
            
            %RGBhistogram  2189-2388
            temp = zeros(1,200);
            h = sum(hist(imgg, 50),2)';
            su = sum(h);
            temp(1,1:50) = h/su;
            
            h = sum(hist(img(:,:,1), 50),2)';
            su = sum(h);
            temp(1,51:100) = h/su;
            
            h = sum(hist(img(:,:,2), 50),2)';
            su = sum(h);
            temp(1,101:150) = h/su;
            
            h = sum(hist(img(:,:,3), 50),2)';
            su = sum(h);
            temp(1,151:200) = h/su;
            mapping(i,2189:2388) = temp;
            
            %region properties
            [h,w,l] = size(imggg)
            diagonale = numel(diag(imggg));
            regprops = regionprops(imggg,'all');
            elements = numel(regprops);
            %number of regions from BW level image
            mapping(i,2389) = double(elements)/(h*w);
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
            su = sum(h);
            mapping(i,2390:2439) = h/su;
            h = hist(cw, 50);
            su = sum(h);
            mapping(i,2440:2489) = h/su;
            h = hist(ch, 50);
            su = sum(h);
            mapping(i,2490:2539) = h/su;
            h = hist(bb, 50);
            su = sum(h);
            mapping(i,2540:2589) = h/su;
            h = hist(mal, 50);
            su = sum(h);
            mapping(i,2590:2639) = h/su;
            h = hist(mil, 50);
            su = sum(h);
            mapping(i,2640:2689) = h/su;
            h = hist(ecc, 50);
            su = sum(h);
            mapping(i,2690:2739) = h/su;
            h = hist(o, 50);
            su = sum(h);
            mapping(i,2740:2789) = h/su;
            h = hist(cva, 50);
            su = sum(h);
            mapping(i,2790:2839) = h/su;
            h = hist(sol, 50);
            su = sum(h);
            mapping(i,2840:2889) = h/su;
            h = hist(ext, 50);
            su = sum(h);
            mapping(i,2890:2939) = h/su;
            
            %region properties
            [h,w,l] = size(imgbw)
            diagonale = numel(diag(imgbw));
            regprops = regionprops(imgbw,'all');
            elements = numel(regprops);
            %number of regions from grey level image
            mapping(i,2940) = double(elements)/(h*w);
            %additional region statistics
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
            su = sum(h);
            mapping(i,2941:2990) = h/su;
            h = hist(cw, 50);
            su = sum(h);
            mapping(i,2991:3040) = h/su;
            h = hist(ch, 50);
            su = sum(h);
            mapping(i,3041:3090) = h/su;
            h = hist(bb, 50);
            su = sum(h);
            mapping(i,3091:3140) = h/su;
            h = hist(mal, 50);
            su = sum(h);
            mapping(i,3141:3190) = h/su;
            h = hist(mil, 50);
            su = sum(h);
            mapping(i,3191:3240) = h/su;
            h = hist(ecc, 50);
            su = sum(h);
            mapping(i,3241:3290) = h/su;
            h = hist(o, 50);
            su = sum(h);
            mapping(i,3291:3340) = h/su;
            h = hist(cva, 50);
            su = sum(h);
            mapping(i,3341:3390) = h/su;
            h = hist(sol, 50);
            su = sum(h);
            mapping(i,3391:3440) = h/su;
            h = hist(ext, 50);
            su = sum(h);
            mapping(i,3441:3490) = h/su;
            
            count = 3491;
            %statistical properties of each features
            %histogram
            mapping(i,count) = std(mapping(i,7:106));
            mapping(i,count+1) = mean(mapping(i,7:106));
            mapping(i,count+2) = var(mapping(i,7:106));
            mapping(i,count+3) = cov(mapping(i,7:106));
            mapping(i,count+4) = median(mapping(i,7:106));
            count = count + 5;
            %FFT
            mapping(i,count) = std(mapping(i,107:206));
            mapping(i,count+1) = mean(mapping(i,107:206));
            mapping(i,count+2) = var(mapping(i,107:206));
            mapping(i,count+3) = cov(mapping(i,107:206));
            mapping(i,count+4) = median(mapping(i,107:206));
            count = count + 5;
            %Gabor
            mapping(i,count) = std(mapping(i,207:306));
            mapping(i,count+1) = mean(mapping(i,207:306));
            mapping(i,count+2) = var(mapping(i,207:306));
            mapping(i,count+3) = cov(mapping(i,207:306));
            mapping(i,count+4) = median(mapping(i,207:306));
            count = count + 5;
            
            mapping(i,count) = std(mapping(i,307:406));
            mapping(i,count+1) = mean(mapping(i,307:406));
            mapping(i,count+2) = var(mapping(i,307:406));
            mapping(i,count+3) = cov(mapping(i,307:406));
            mapping(i,count+4) = median(mapping(i,307:406));
            count = count + 5;
            
            mapping(i,count) = std(mapping(i,407:506));
            mapping(i,count+1) = mean(mapping(i,407:506));
            mapping(i,count+2) = var(mapping(i,407:506));
            mapping(i,count+3) = cov(mapping(i,407:506));
            mapping(i,count+4) = median(mapping(i,407:506));
            count = count + 5;
            
            mapping(i,count) = std(mapping(i,507:606));
            mapping(i,count+1) = mean(mapping(i,507:606));
            mapping(i,count+2) = var(mapping(i,507:606));
            mapping(i,count+3) = cov(mapping(i,507:606));
            mapping(i,count+4) = median(mapping(i,507:606));
            count = count + 5;
            
            mapping(i,count) = std(mapping(i,607:706));
            mapping(i,count+1) = mean(mapping(i,607:706));
            mapping(i,count+2) = var(mapping(i,607:706));
            mapping(i,count+3) = cov(mapping(i,607:706));
            mapping(i,count+4) = median(mapping(i,607:706));
            count = count + 5;
            %Gist
            mapping(i,count) = std(mapping(i,709:1668));
            mapping(i,count+1) = mean(mapping(i,709:1668));
            mapping(i,count+2) = var(mapping(i,709:1668));
            mapping(i,count+3) = cov(mapping(i,709:1668));
            mapping(i,count+4) = median(mapping(i,709:1668));
            count = count + 5;
            
            %3602 to 3722 statistical data about HOG
            
            %line feature
            mapping(i,count) = std(mapping(1,2069:2188));
            mapping(i,count+1) = mean(mapping(1,2069:2188));
            mapping(i,count+2) = var(mapping(1,2069:2188));
            mapping(i,count+3) = cov(mapping(1,2069:2188));
            mapping(i,count+4) = median(mapping(1,2069:2188));
            count = count + 5;
            
            %RGBhistogram
            mapping(i,count) = std(mapping(1,2189:2388));
            mapping(i,count+1) = mean(mapping(1,2189:2388));
            mapping(i,count+2) = var(mapping(1,2189:2388));
            mapping(i,count+3) = cov(mapping(1,2189:2388));
            mapping(i,count+4) = median(mapping(1,2189:2388));
            count = count + 5;
            
            %BW region statistics
            mapping(i,count) = std(mapping(1,2390:2439));
            mapping(i,count+1) = mean(mapping(1,2390:2439));
            mapping(i,count+2) = var(mapping(1,2390:2439));
            mapping(i,count+3) = cov(mapping(1,2390:2439));
            mapping(i,count+4) = median(mapping(1,2390:2439));
            count = count + 5;
            
            mapping(i,count) = std(mapping(1,2440:2489));
            mapping(i,count+1) = mean(mapping(1,2440:2489));
            mapping(i,count+2) = var(mapping(1,2440:2489));
            mapping(i,count+3) = cov(mapping(1,2440:2489));
            mapping(i,count+4) = median(mapping(1,2440:2489));
            count = count + 5;
            
            mapping(i,count) = std(mapping(1,2490:2539));
            mapping(i,count+1) = mean(mapping(1,2490:2539));
            mapping(i,count+2) = var(mapping(1,2490:2539));
            mapping(i,count+3) = cov(mapping(1,2490:2539));
            mapping(i,count+4) = median(mapping(1,2490:2539));
            count = count + 5;
            
            mapping(i,count) = std(mapping(1,2540:2589));
            mapping(i,count+1) = mean(mapping(1,2540:2589));
            mapping(i,count+2) = var(mapping(1,2540:2589));
            mapping(i,count+3) = cov(mapping(1,2540:2589));
            mapping(i,count+4) = median(mapping(1,2540:2589));
            count = count + 5;
            
            mapping(i,count) = std(mapping(1,2590:2639));
            mapping(i,count+1) = mean(mapping(1,2590:2639));
            mapping(i,count+2) = var(mapping(1,2590:2639));
            mapping(i,count+3) = cov(mapping(1,2590:2639));
            mapping(i,count+4) = median(mapping(1,2590:2639));
            count = count + 5;
            
            mapping(i,count) = std(mapping(1,2640:2689));
            mapping(i,count+1) = mean(mapping(1,2640:2689));
            mapping(i,count+2) = var(mapping(1,2640:2689));
            mapping(i,count+3) = cov(mapping(1,2640:2689));
            mapping(i,count+4) = median(mapping(1,2640:2689));
            count = count + 5;
            
            mapping(i,count) = std(mapping(1,2690:2739));
            mapping(i,count+1) = mean(mapping(1,2690:2739));
            mapping(i,count+2) = var(mapping(1,2690:2739));
            mapping(i,count+3) = cov(mapping(1,2690:2739));
            mapping(i,count+4) = median(mapping(1,2690:2739));
            count = count + 5;
            
            mapping(i,count) = std(mapping(1,2740:2789));
            mapping(i,count+1) = mean(mapping(1,2740:2789));
            mapping(i,count+2) = var(mapping(1,2740:2789));
            mapping(i,count+3) = cov(mapping(1,2740:2789));
            mapping(i,count+4) = median(mapping(1,2740:2789));
            count = count + 5;
            
            mapping(i,count) = std(mapping(1,2790:2839));
            mapping(i,count+1) = mean(mapping(1,2790:2839));
            mapping(i,count+2) = var(mapping(1,2790:2839));
            mapping(i,count+3) = cov(mapping(1,2790:2839));
            mapping(i,count+4) = median(mapping(1,2790:2839));
            count = count + 5;
            
            mapping(i,count) = std(mapping(1,2840:2889));
            mapping(i,count+1) = mean(mapping(1,2840:2889));
            mapping(i,count+2) = var(mapping(1,2840:2889));
            mapping(i,count+3) = cov(mapping(1,2840:2889));
            mapping(i,count+4) = median(mapping(1,2840:2889));
            count = count + 5;
            
            mapping(i,count) = std(mapping(1,2890:2939));
            mapping(i,count+1) = mean(mapping(1,2890:2939));
            mapping(i,count+2) = var(mapping(1,2890:2939));
            mapping(i,count+3) = cov(mapping(1,2890:2939));
            mapping(i,count+4) = median(mapping(1,2890:2939));
            count = count + 5;
            
            %Gray Image region statistics
            mapping(i,count) = std(mapping(1,2940:2989));
            mapping(i,count+1) = mean(mapping(1,2940:2989));
            mapping(i,count+2) = var(mapping(1,2940:2989));
            mapping(i,count+3) = cov(mapping(1,2940:2989));
            mapping(i,count+4) = median(mapping(1,2940:2989));
            count = count + 5;
            
            mapping(i,count) = std(mapping(1,2990:3039));
            mapping(i,count+1) = mean(mapping(1,2990:3039));
            mapping(i,count+2) = var(mapping(1,2990:3039));
            mapping(i,count+3) = cov(mapping(1,2990:3039));
            mapping(i,count+4) = median(mapping(1,2990:3039));
            count = count + 5;
            
            mapping(i,count) = std(mapping(1,3040:3089));
            mapping(i,count+1) = mean(mapping(1,3040:3089));
            mapping(i,count+2) = var(mapping(1,3040:3089));
            mapping(i,count+3) = cov(mapping(1,3040:3089));
            mapping(i,count+4) = median(mapping(1,3040:3089));
            count = count + 5;
            
            mapping(i,count) = std(mapping(1,3090:3139));
            mapping(i,count+1) = mean(mapping(1,3090:3139));
            mapping(i,count+2) = var(mapping(1,3090:3139));
            mapping(i,count+3) = cov(mapping(1,3090:3139));
            mapping(i,count+4) = median(mapping(1,3090:3139));
            count = count + 5;
            
            mapping(i,count) = std(mapping(1,3140:3189));
            mapping(i,count+1) = mean(mapping(1,3140:3189));
            mapping(i,count+2) = var(mapping(1,3140:3189));
            mapping(i,count+3) = cov(mapping(1,3140:3189));
            mapping(i,count+4) = median(mapping(1,3140:3189));
            count = count + 5;
            
            mapping(i,count) = std(mapping(1,3190:3239));
            mapping(i,count+1) = mean(mapping(1,3190:3239));
            mapping(i,count+2) = var(mapping(1,3190:3239));
            mapping(i,count+3) = cov(mapping(1,3190:3239));
            mapping(i,count+4) = median(mapping(1,3190:3239));
            count = count + 5;
            
            mapping(i,count) = std(mapping(1,3240:3289));
            mapping(i,count+1) = mean(mapping(1,3240:3289));
            mapping(i,count+2) = var(mapping(1,3240:3289));
            mapping(i,count+3) = cov(mapping(1,3240:3289));
            mapping(i,count+4) = median(mapping(1,3240:3289));
            count = count + 5;
            
            mapping(i,count) = std(mapping(1,3290:3339));
            mapping(i,count+1) = mean(mapping(1,3290:3339));
            mapping(i,count+2) = var(mapping(1,3290:3339));
            mapping(i,count+3) = cov(mapping(1,3290:3339));
            mapping(i,count+4) = median(mapping(1,3290:3339));
            count = count + 5;
            
            mapping(i,count) = std(mapping(1,3340:3389));
            mapping(i,count+1) = mean(mapping(1,3340:3389));
            mapping(i,count+2) = var(mapping(1,3340:3389));
            mapping(i,count+3) = cov(mapping(1,3340:3389));
            mapping(i,count+4) = median(mapping(1,3340:3389));
            count = count + 5;
            
            mapping(i,count) = std(mapping(1,3390:3439));
            mapping(i,count+1) = mean(mapping(1,3390:3439));
            mapping(i,count+2) = var(mapping(1,3390:3439));
            mapping(i,count+3) = cov(mapping(1,3390:3439));
            mapping(i,count+4) = median(mapping(1,3390:3439));
            count = count + 5;
            
            mapping(i,count) = std(mapping(1,3440:3489));
            mapping(i,count+1) = mean(mapping(1,3440:3489));
            mapping(i,count+2) = var(mapping(1,3440:3489));
            mapping(i,count+3) = cov(mapping(1,3440:3489));
            mapping(i,count+4) = median(mapping(1,3440:3489));
            count = count + 5;
        end
    end;
end

save(strcat('%s/Features_Whole_Images_0%s',path,'.mat'),'mapping');
