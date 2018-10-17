function [ result] = find_best_features_normalized_single(year, ids, j, ii, iis, iie)
%computes features for images - region images 

%file_pattern='\d\d\d\d_\d\d\d\d\d\d.jpg';
mapping = [];%zeros(1,9036);

%size(ii)
%size(iis)
%size(iie)

%prepare the images from which to compute the features
iig = rgb2gray(ii);
ibw = im2bw(ii);
imbw = im2double(ibw);
imgbw = imbw;
img = im2double(ii);
imgg = im2double(iig);
imggg = iig;
%[h,w] = size(iig);

 sprintf('processing...')

count = 1;
%[lh, f] = his(imgg);
%mapping = [mapping,f];
%%mapping(1, count:lh(2)) = f;
%count = count + lh(2);
%[lf, f] = ft(imgg);
%mapping = [mapping,f];
%%mapping(1, count:count+lf(2)-1) = f;
%count = count + lf(2);
%[lg, f] = gab(imgg);
%mapping = [mapping,f];
%%mapping(1, count:count+lg(2)-1) = f;
%count = count + lg(2);
%[lw, f] = wav(imgg);
%mapping = [mapping,f];
%%mapping(1, count:count+lw(2)-1) = f;
%count = count + lw(2);
%[lc, f] = con(imgg);
%mapping = [mapping,f];
%%mapping(1, count:count+lc(2)-1) = f;
%count = count + lc(2);
%[la, f] = acu(imgg);
%mapping = [mapping,f];
%%mapping(1, count:count+la(2)-1) = f;
%count = count + la(2);
%[li, f] = gis(img);
%mapping = [mapping,f];
%%mapping(1, count:count+li(2)-1) = f;
%count = count + li(2);
%[lo, f] = ogr(img);
%mapping = [mapping,f];
%%mapping(1, count:count+lo(2)-1) = f;
%count = count + lo(2);
%[ls, f] = lsf(iis, iie);
%mapping = [mapping,f];
%%mapping(1, count:count+ls(2)-1) = f;
%count = count + ls(2);
%[ll, f] = lif(iis, iie);
%mapping = [mapping,f];
%%mapping(1, count:count+ll(2)-1) = f;
%count = count + ll(2);
%[llg, f] = lgf(iis, iie);
%mapping = [mapping,f];
%%mapping(1, count:count+llg(2)-1) = f;
%count = count + llg(2);
%[lb, f] = rgb(imgg, img);
%mapping = [mapping,f];
%mapping(1, count:count+lb(2)-1) = f;
%count = count + lb(2);
%[lr, f] = rgf(imggg);
%mapping = [mapping,f];
%%mapping(1, count:count+lr(2)-1) = f;
%count = count + lr(2);
%[lrr, f] = rbf (imgbw);
%mapping = [mapping,f];
%%mapping(1, count:count+lrr(2)-1) = f;
%count = count + lrr(2);
%[attr, f] = attribs (imggg);
%mapping = [mapping,f];
[attr, f0] = attribs_no_hist(imggg);
%[attr, f1] = attribs_hist(imggg);
mapping = [mapping,f0];
%mapping = [mapping,f1];
%mapping(1, count:count+attr(2)-1) = f;
count = count + attr(2);
%     counter = counter + lh;
%     counter = counter + lf;
%     counter = counter + lg;
%     counter = counter + lw;
%     counter = counter + lc;
%     counter = counter + la;
%     counter = counter + li;
%     counter = counter + lo;
%     counter = counter + ls;
%     counter = counter + ll;
%     counter = counter + llg;
%     counter = counter + lb;
%     counter = counter + lr;
%     counter = counter + lrr;
%size(mapping);

%result = zeros(1,count+5);
result = zeros(attr(1),count+2);
%size(result);
result(:,4:end) = mapping;
result(:,1) = year;
result(:,2) = ids;
result(:,3) = j;

%mapping = mapping(1,1:counter+1);
%save(strcat(dir_path,'/Features_Whole_Images_Normalized.mat'),'mapping');
end

function [l, f] = his(imgg)
            %sum of gray histogram
            H = imhist(imgg, 100); %min-max
            f = zeros(1,105);
            %normalize
            %s = size(H);
            su = sum(H(:));%/s(2);
            f(1,1:100) = double(H)./su;
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
            mags = sum(abs(FH));%/100;
            reals = real(FH)/mags;
            imags = imag(FH)/mags;
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
            [~,gabout] = gaborfilter1(imgg,2,4,16,4*pi/3);
            GH = imhist(im2double(gabout),100);
            %GH = sum(histo');
            %s = size(GH);
            su = sum(GH(:));%/s(2);
            f = zeros(1,105);
            f(1,1:100) = double(GH)./su;
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
            %s = size(CH);
            su = sum(CH(:));%/s(2);
            ch = double(CH)./su;
            f(1,1:100) = ch;
            %stats
            f(1,101) = std(f(1,1:100));
            f(1,102) = mean(f(1,1:100));
            f(1,103) = var(f(1,1:100));
            f(1,104) = cov(f(1,1:100));
            f(1,105) = median(f(1,1:100));
            
            %s = size(CA);
            su = sum(CA(:));%/s(2);
            ca = double(CA)./su;
            f(1,106:205) = ca;
            %stats
            f(1,206) = std(f(1,106:205));
            f(1,207) = mean(f(1,106:205));
            f(1,208) = var(f(1,106:205));
            f(1,209) = cov(f(1,106:205));
            f(1,210) = median(f(1,106:205));
            
            %s = size(CV);
            su = sum(CV(:));%/s(2);
            cv = double(CV)./su;
            f(1,211:310) = cv;
            %stats
            f(1,311) = std(f(1,211:310));
            f(1,312) = mean(f(1,211:310));
            f(1,313) = var(f(1,211:310));
            f(1,314) = cov(f(1,211:310));
            f(1,315) = median(f(1,211:310));
            
            %s = size(CD);
            su = sum(CD(:));%/s(2);
            cd = double(CD)./su;
            f(1,316:415) = cd;
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
            %[h,w,l] = size(img);
            %A = min(h,w);
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
            [BGa,~] = detBG(img);
            [TGa,~] = detTG(img);
            [CGa,~] = detCG(img);
            f = zeros(1,415);
            cou = 1;
            for dr = 1:8,
                BGA = imhist(BGa(:,:,dr),128);
                %s = size(BGA);
                su = sum(BGA(:));%/s(2);
                bga = double(BGA)./su;
                f(1,cou:cou+127) = bga;
                cou = cou + 128;
                %stats
                f(1,cou) = std(bga);
                f(1,cou+1) = mean(bga);
                f(1,cou+2) = var(bga);
                f(1,cou+3) = cov(bga);
                f(1,cou+4) = median(bga);
                cou = cou + 5;
                
                TGA = imhist(TGa(:,:,dr),128);
                %s = size(TGA);
                su = sum(TGA(:));%/s(2);
                tga = double(TGA)./su;
                f(1,cou:cou+127) = tga;
                cou = cou + 128;
                %stats
                f(1,cou) = std(tga);
                f(1,cou+1) = mean(tga);
                f(1,cou+2) = var(tga);
                f(1,cou+3) = cov(tga);
                f(1,cou+4) = median(tga);
                cou = cou + 5;
                
                for color = 1:3,
                    COLRS = imhist(CGa(:,:,color,dr),128);
                    %s = size(COLRS);
                    su = sum(COLRS(:));%/s(2);
                    colrs = double(COLRS)./su;
                    f(1,cou:cou+127) = colrs;
                    cou = cou + 128;
                    %stats
                    f(1,cou) = std(colrs);
                    f(1,cou+1) = mean(colrs);
                    f(1,cou+2) = var(colrs);
                    f(1,cou+3) = cov(colrs);
                    f(1,cou+4) = median(colrs);
                    cou = cou + 5;
                end
            end            
            l = size(f);
end
function [l, f] = lsf(iis, iie)            
            %line feature 2069-2188 from Saliency 1
            f = zeros(1,133);
            [m,n,~] = size(iis);
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
            %yy = offset   + linecou
            su = sum(f(1,1:end));%/s;
            lins = double(f(1,1:127))./su;
%            size(lins)
            f(1,1:127) = lins;
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
            [m,n,~] = size(iis);
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
            
%            s = 2307-2189+1;
            su = sum(f(1,1:end));%/s;
            lins = double(f(1,1:end))./su;
            f(1,1:end) = lins;
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
            [m,n,~] = size(iis);
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
            
%            s = 2426-2308+1;
            su = sum(f(1,1:end));%/s;
            lins = double(f(1,1:end))./su;
            f(1,1:end) = lins;
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
            h = sum(hist(imgg, 100),2)';
            su = sum(h);
            f(1,1:100) = h/su;
            %stats
            f(1,101) = std(f(1,1:100));
            f(1,102) = mean(f(1,1:100));
            f(1,103) = var(f(1,1:100));
            f(1,104) = cov(f(1,1:100));
            f(1,105) = median(f(1,1:100));
            
            h = sum(hist(img(:,:,1), 100),2)';
            su = sum(h);
            f(1,106:205) = h/su;
            %stats
            f(1,206) = std(f(1,106:205));
            f(1,207) = mean(f(1,106:205));
            f(1,208) = var(f(1,106:205));
            f(1,209) = cov(f(1,106:205));
            f(1,210) = median(f(1,106:205));
            
            h = sum(hist(img(:,:,2), 100),2)';
            su = sum(h);
            f(1,211:310) = h/su;
            %stats
            f(1,311) = std(f(1,211:310));
            f(1,312) = mean(f(1,211:310));
            f(1,313) = var(f(1,211:310));
            f(1,314) = cov(f(1,211:310));
            f(1,315) = median(f(1,211:310));
            
            h = sum(hist(img(:,:,3), 100),2)';
            su = sum(h);
            f(1,316:415) = h/su;
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
            [h,w,~] = size(imggg);
            diagonale = numel(diag(imggg));
            regprops = regionprops(imggg,'all');
            elements = numel(regprops);
            %number of regions from BW level image
            f = zeros(1,605);
            f(1,1) = double(elements)/(h*w);
            %additional region statistics
            %number of regions
            %mapping(i,data) = elements;
            Area = zeros(1, elements);
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
                Area(u) = double(regprops(u).Area)/(h*w);
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
            h = hist(Area, 50);
            su = sum(h);
            f(1,2:51) = h/su;
            %stats
            f(1,52) = std(f(1,2:51));
            f(1,53) = mean(f(1,2:51));
            f(1,54) = var(f(1,2:51));
            f(1,55) = cov(f(1,2:51));
            f(1,56) = median(f(1,2:51));           
            h = hist(cw, 50);
            su = sum(h);
            f(1,57:106) = h/su;
            %stats
            f(1,107) = std(f(1,57:106));
            f(1,108) = mean(f(1,57:106));
            f(1,109) = var(f(1,57:106));
            f(1,110) = cov(f(1,57:106));
            f(1,111) = median(f(1,57:106));
            h = hist(ch, 50);
            su = sum(h);
            f(1,112:161) = h/su;
            %stats
            f(1,162) = std(f(1,112:161));
            f(1,163) = mean(f(1,112:161));
            f(1,164) = var(f(1,112:161));
            f(1,165) = cov(f(1,112:161));
            f(1,166) = median(f(1,112:161));
            h = hist(bb, 50);
            su = sum(h);
            f(1,166:215) = h/su;
            %stats
            f(1,216) = std(f(1,166:215));
            f(1,217) = mean(f(1,166:215));
            f(1,218) = var(f(1,166:215));
            f(1,219) = cov(f(1,166:215));
            f(1,220) = median(f(1,166:215));
            h = hist(mal, 50);
            su = sum(h);
            f(1,221:270) = h/su;
            %stats
            f(1,271) = std(f(1,221:270));
            f(1,272) = mean(f(1,221:270));
            f(1,273) = var(f(1,221:270));
            f(1,274) = cov(f(1,221:270));
            f(1,275) = median(f(1,221:270));
            h = hist(mil, 50);
            su = sum(h);
            f(1,276:325) = h/su;
            %stats
            f(1,326) = std(f(1,276:325));
            f(1,327) = mean(f(1,276:325));
            f(1,328) = var(f(1,276:325));
            f(1,329) = cov(f(1,276:325));
            f(1,330) = median(f(1,276:325));
            h = hist(ecc, 50);
            su = sum(h);
            f(1,331:380) = h/su;
            %stats
            f(1,381) = std(f(1,331:380));
            f(1,382) = mean(f(1,331:380));
            f(1,383) = var(f(1,331:380));
            f(1,384) = cov(f(1,331:380));
            f(1,385) = median(f(1,331:380));
            h = hist(o, 50);
            su = sum(h);
            f(1,386:435) = h/su;
            %stats
            f(1,436) = std(f(1,386:435));
            f(1,437) = mean(f(1,386:435));
            f(1,438) = var(f(1,386:435));
            f(1,439) = cov(f(1,386:435));
            f(1,440) = median(f(1,386:435));
            h = hist(cva, 50);
            su = sum(h);
            f(1,441:490) = h/su;
            %stats
            f(1,491) = std(f(1,441:490));
            f(1,492) = mean(f(1,441:490));
            f(1,493) = var(f(1,441:490));
            f(1,494) = cov(f(1,441:490));
            f(1,495) = median(f(1,441:490));
            h = hist(sol, 50);
            su = sum(h);
            f(1,496:545) = h/su;
            %stats
            f(1,546) = std(f(1,496:545));
            f(1,547) = mean(f(1,496:545));
            f(1,548) = var(f(1,496:545));
            f(1,549) = cov(f(1,496:545));
            f(1,550) = median(f(1,496:545));
            h = hist(ext, 50);
            su = sum(h);
            f(1,551:600) = h/su;
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
            [h,w,~] = size(imgbw);
            f = zeros(1,605);
            diagonale = numel(diag(imgbw));
            regprops = regionprops(imgbw,'all');
            elements = numel(regprops);
            %number of regions from grey level image
            f(1,1) = double(elements)/(h*w);
            %additional region statistics
            Area = zeros(1, elements);
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
                Area(u) = double(regprops(u).Area)/(h*w);
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
            h = hist(Area, 50);
            su = sum(h);
            f(1,2:51) = h/su;
            %stats
            f(1,52) = std(f(1,2:51));
            f(1,53) = mean(f(1,2:51));
            f(1,54) = var(f(1,2:51));
            f(1,55) = cov(f(1,2:51));
            f(1,56) = median(f(1,2:51));           
            h = hist(cw, 50);
            su = sum(h);
            f(1,57:106) = h/su;
            %stats
            f(1,107) = std(f(1,57:106));
            f(1,108) = mean(f(1,57:106));
            f(1,109) = var(f(1,57:106));
            f(1,110) = cov(f(1,57:106));
            f(1,111) = median(f(1,57:106));
            h = hist(ch, 50);
            su = sum(h);
            f(1,112:161) = h/su;
            %stats
            f(1,162) = std(f(1,112:161));
            f(1,163) = mean(f(1,112:161));
            f(1,164) = var(f(1,112:161));
            f(1,165) = cov(f(1,112:161));
            f(1,166) = median(f(1,112:161));
            h = hist(bb, 50);
            su = sum(h);
            f(1,166:215) = h/su;
            %stats
            f(1,216) = std(f(1,166:215));
            f(1,217) = mean(f(1,166:215));
            f(1,218) = var(f(1,166:215));
            f(1,219) = cov(f(1,166:215));
            f(1,220) = median(f(1,166:215));
            h = hist(mal, 50);
            su = sum(h);
            f(1,221:270) = h/su;
            %stats
            f(1,271) = std(f(1,221:270));
            f(1,272) = mean(f(1,221:270));
            f(1,273) = var(f(1,221:270));
            f(1,274) = cov(f(1,221:270));
            f(1,275) = median(f(1,221:270));
            h = hist(mil, 50);
            su = sum(h);
            f(1,276:325) = h/su;
            %stats
            f(1,326) = std(f(1,276:325));
            f(1,327) = mean(f(1,276:325));
            f(1,328) = var(f(1,276:325));
            f(1,329) = cov(f(1,276:325));
            f(1,330) = median(f(1,276:325));
            h = hist(ecc, 50);
            su = sum(h);
            f(1,331:380) = h/su;
            %stats
            f(1,381) = std(f(1,331:380));
            f(1,382) = mean(f(1,331:380));
            f(1,383) = var(f(1,331:380));
            f(1,384) = cov(f(1,331:380));
            f(1,385) = median(f(1,331:380));
            h = hist(o, 50);
            su = sum(h);
            f(1,386:435) = h/su;
            %stats
            f(1,436) = std(f(1,386:435));
            f(1,437) = mean(f(1,386:435));
            f(1,438) = var(f(1,386:435));
            f(1,439) = cov(f(1,386:435));
            f(1,440) = median(f(1,386:435));
            h = hist(cva, 50);
            su = sum(h);
            f(1,441:490) = h/su;
            %stats
            f(1,491) = std(f(1,441:490));
            f(1,492) = mean(f(1,441:490));
            f(1,493) = var(f(1,441:490));
            f(1,494) = cov(f(1,441:490));
            f(1,495) = median(f(1,441:490));
            h = hist(sol, 50);
            su = sum(h);
            f(1,496:545) = h/su;
            %stats
            f(1,546) = std(f(1,496:545));
            f(1,547) = mean(f(1,496:545));
            f(1,548) = var(f(1,496:545));
            f(1,549) = cov(f(1,496:545));
            f(1,550) = median(f(1,496:545));
            h = hist(ext, 50);
            su = sum(h);
            f(1,551:600) = h/su;
            %stats
            f(1,601) = std(f(1,551:600));
            f(1,602) = mean(f(1,551:600));
            f(1,603) = var(f(1,551:600));
            f(1,604) = cov(f(1,551:600));
            f(1,605) = median(f(1,551:600));
            l= size(f);

end

function [l, f] = attribs(imggg)
            %region properties
            [h,w,~] = size(imggg);
            diagonale = numel(diag(imggg));
            regprops = regionprops(imggg,'all');
            elements = numel(regprops);
            %number of regions from BW level image
            f = zeros(1,716);
            f(1,1) = double(elements)/(h*w);
            %additional region statistics
            %number of regions
            %mapping(i,data) = elements;
            Area = zeros(1, elements);
            cw = zeros(1, elements);
            ch = zeros(1, elements);
            bb = zeros(1, elements);
            mal = zeros(1, elements);
            mil = zeros(1, elements);
            malmil = zeros(1, elements);
            ecc = zeros(1, elements);
            o = zeros(1, elements);
            eul = zeros(1, elements);
            cva = zeros(1, elements);
            sol = zeros(1, elements);
            ext = zeros(1, elements);
            for u=1:elements,
                Area(u) = double(regprops(u).Area)/(h*w);
                cw(u) = double(regprops(u).Centroid(1))/w;
                ch(u) = double(regprops(u).Centroid(2))/h;
                bb(u) = double(regprops(u).BoundingBox(3)*regprops(u).BoundingBox(4))/(h*w);
                cva(u) = double(regprops(u).ConvexArea)/(h*w);
                ecc(u) = regprops(u).Eccentricity;
                eul(u) = double(regprops(u).EulerNumber);
                ext(u) = regprops(u).Extent;
                mal(u) = double(regprops(u).MajorAxisLength)/diagonale;
                mil(u) = double(regprops(u).MinorAxisLength)/diagonale;
                malmil(u) = double(regprops(u).MinorAxisLength)/double(regprops(u).MajorAxisLength);
                o(u) = double(regprops(u).Orientation);
                sol(u) = regprops(u).Solidity;
            end
            h = hist(Area, 50);
            su = sum(h);
            f(1,2:51) = h/su;
            %stats
            f(1,52) = std(f(1,2:51));
            f(1,53) = mean(f(1,2:51));
            f(1,54) = var(f(1,2:51));
            f(1,55) = cov(f(1,2:51));
            f(1,56) = median(f(1,2:51));           
            
            h = hist(cw, 50);
            su = sum(h);
            f(1,57:106) = h/su;
            %stats
            f(1,107) = std(f(1,57:106));
            f(1,108) = mean(f(1,57:106));
            f(1,109) = var(f(1,57:106));
            f(1,110) = cov(f(1,57:106));
            f(1,111) = median(f(1,57:106));
            
            h = hist(ch, 50);
            su = sum(h);
            f(1,112:161) = h/su;
            %stats
            f(1,162) = std(f(1,112:161));
            f(1,163) = mean(f(1,112:161));
            f(1,164) = var(f(1,112:161));
            f(1,165) = cov(f(1,112:161));
            f(1,166) = median(f(1,112:161));
            
            h = hist(bb, 50);
            su = sum(h);
            f(1,167:216) = h/su;
            %stats
            f(1,217) = std(f(1,166:215));
            f(1,218) = mean(f(1,166:215));
            f(1,219) = var(f(1,166:215));
            f(1,220) = cov(f(1,166:215));
            f(1,221) = median(f(1,166:215));
            
            h = hist(cva, 50);
            su = sum(h);
            f(1,222:271) = h/su;
            %stats
            f(1,272) = std(f(1,441:490));
            f(1,273) = mean(f(1,441:490));
            f(1,274) = var(f(1,441:490));
            f(1,275) = cov(f(1,441:490));
            f(1,276) = median(f(1,441:490));
            
            h = hist(ecc, 50);
            su = sum(h);
            f(1,277:326) = h/su;
            %stats
            f(1,327) = std(f(1,331:380));
            f(1,328) = mean(f(1,331:380));
            f(1,329) = var(f(1,331:380));
            f(1,330) = cov(f(1,331:380));
            f(1,331) = median(f(1,331:380));
            
            h = hist(eul, 50);
            su = sum(h);
            f(1,332:381) = h/su;
            %stats
            f(1,382) = std(f(1,331:380));
            f(1,383) = mean(f(1,331:380));
            f(1,384) = var(f(1,331:380));
            f(1,385) = cov(f(1,331:380));
            f(1,386) = median(f(1,331:380));

            h = hist(ext, 50);
            su = sum(h);
            f(1,387:436) = h/su;
            %stats
            f(1,437) = std(f(1,551:600));
            f(1,438) = mean(f(1,551:600));
            f(1,439) = var(f(1,551:600));
            f(1,440) = cov(f(1,551:600));
            f(1,441) = median(f(1,551:600));
            
            h = hist(mal, 50);
            su = sum(h);
            f(1,442:491) = h/su;
            %stats
            f(1,492) = std(f(1,221:270));
            f(1,493) = mean(f(1,221:270));
            f(1,494) = var(f(1,221:270));
            f(1,495) = cov(f(1,221:270));
            f(1,496) = median(f(1,221:270));

            h = hist(mil, 50);
            su = sum(h);
            f(1,497:546) = h/su;
            %stats
            f(1,547) = std(f(1,276:325));
            f(1,548) = mean(f(1,276:325));
            f(1,549) = var(f(1,276:325));
            f(1,550) = cov(f(1,276:325));
            f(1,551) = median(f(1,276:325));

            h = hist(malmil, 50);
            su = sum(h);
            f(1,552:601) = h/su;
            %stats
            f(1,602) = std(f(1,276:325));
            f(1,603) = mean(f(1,276:325));
            f(1,604) = var(f(1,276:325));
            f(1,605) = cov(f(1,276:325));
            f(1,606) = median(f(1,276:325));

            h = hist(o, 50);
            su = sum(h);
            f(1,607:656) = h/su;
            %stats
            f(1,657) = std(f(1,386:435));
            f(1,658) = mean(f(1,386:435));
            f(1,659) = var(f(1,386:435));
            f(1,660) = cov(f(1,386:435));
            f(1,661) = median(f(1,386:435));
            
            h = hist(sol, 50);
            su = sum(h);
            f(1,662:711) = h/su;
            %stats
            f(1,712) = std(f(1,496:545));
            f(1,713) = mean(f(1,496:545));
            f(1,714) = var(f(1,496:545));
            f(1,715) = cov(f(1,496:545));
            f(1,716) = median(f(1,496:545));

            l = size(f);
end

function [l, f] = attribs_no_hist(imggg)
            %region properties
            [h,w,~] = size(imggg);
            diagonale = numel(diag(imggg));
            regprops = regionprops(imggg,'all');
            [~,idx]=sort([regprops.Area],'descend');
            regprops=regprops(idx);
            elements = numel(regprops);
            %number of regions from BW level image
            f = zeros(elements,13);
            %additional region statistics
            %number of regions
            %mapping(i,data) = elements;
            for u=1:elements
                f(u,1) = double(regprops(u).Area)/(h*w);
                f(u,2) = double(regprops(u).Centroid(1))/w;
                f(u,3) = double(regprops(u).Centroid(2))/h;
                f(u,4) = double(regprops(u).BoundingBox(3)*regprops(u).BoundingBox(4))/(h*w);
                f(u,5) = double(regprops(u).ConvexArea)/(h*w);
                f(u,6) = regprops(u).Eccentricity;
                f(u,7) = double(regprops(u).EulerNumber);
                f(u,8) = regprops(u).Extent;
                f(u,9) = double(regprops(u).MajorAxisLength)/diagonale;
                f(u,10) = double(regprops(u).MinorAxisLength)/diagonale;
                f(u,11) = double(regprops(u).MinorAxisLength)/double(regprops(u).MajorAxisLength);
                f(u,12) = double(regprops(u).Orientation);
                f(u,13) = regprops(u).Solidity;
            end
		
            l = size(f);
end

