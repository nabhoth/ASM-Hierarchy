function [ mapping] = get_feats(im,mask,mapping,index,counter)
%computes features for images - whole images 
%
%parameters:
%           dir_path - path to the directories containing the images 
%output:
%           mapping - array of features 

            
            %read the image from which to compute the features
            I = im;
            %sum of non zero pixels
%            nonzero = sum(sum(I > 0));

            rgbnonzeroim = I(I > 0);
%            nonzeroimpix = I(I.* repmat(mask,[1,1,3]) > 0);
            
            IG = rgb2gray(I);
            graynonzeroim = IG(IG > 0);

            IBW = im2bw(I);
            imBW = im2double(IBW);
            imgg = im2double(IG);
            [h,w] = size(IG);
            
            
            fprintf('processing...\n');
            count = counter;

            [lh, f] = his(graynonzeroim);
            mapping{index, count} = f(1,1);
            count = count + lh;
            
            [lg, f] = ft(imgg);
            mapping{index, count} = f;
            count = count + lg(2);

            [lg, f] = gab(IG);
            mapping{index, count} = f;
            count = count + lg(2);

            [lb, f] = rgb(rgbnonzeroim);
            mapping{index, count} = f(1,1);
            count = count + lb(2);

            [lr, f] = rgf(IG);
            mapping{index, count} = f(1,1);
            count = count + lr(2);
            
            [lrr, f] = rbf (imBW);
            mapping{index, count} = f;
            count = count + lrr(2);
            
%             [lo, f] = ogr(img);            
%             mapping(index, count:count+lo(2)-1) = f(1,1:40);
%             count = count + lo(2);
            

            
%            mapping(index, 3:counter+2) = mapping(index, 1:counter);
%            mapping(1,1) = str2double(id);
end

function [l, f] = his(imgg)
            %sum of gray histogram
            H = (hist(imgg, 10)'); %min-max
            avg = (abs(H - mean(H)));
            [i,f] = (min(avg));
%            f = round(10*(double(mean(double(H))*10)))/nonzerosum;
%             %stats
%             f(1,101) = std(f(1,1:100));
%             f(1,102) = mean(f(1,1:100));
%             f(1,103) = var(f(1,1:100));
%             f(1,104) = cov(f(1,1:100));
%             f(1,105) = median(f(1,1:100));
            l = 1;
end            
            
function [l, f] = ft(imgg)           
            %sum of fft
            try
            histo = sum(hist(fft(imgg,10),10)');
%             reals = real(FH);
%             imags = imag(FH);
            avg = (abs(histo - mean(histo)));
            [i,f] = (min(avg));
            catch
                f = 0;
            end
%             f = zeros(1,105);
%             f(1,1:100) = complex(reals,imags);
            l = size(f);
end

function [l, f] = gab(imgg)         
            %gabor filter
            [G,gabout] = gaborfilter1(imgg(:,:,1),2,4,16,4*pi/3);
            histo = hist(reshape(gabout, prod(size(gabout)), 1),10);
            avg = (abs(histo - mean(histo)));
            [i,f] = (min(avg));
%            GH = sum(histo');
%            f = (mean(double(GH))*10)/nonzerosum;
            %stats
%             f(1,101) = std(f(1,1:100));
%             f(1,102) = mean(f(1,1:100));
%             f(1,103) = var(f(1,1:100));
%             f(1,104) = cov(f(1,1:100));
%             f(1,105) = median(f(1,1:100));
            l = size(f);
end

function [l, f]=wav(imgg)          
            %wavelet
            [cA,cH,cV,cD] = (dwt2(imgg,'haar'));
            CH = sum(hist(cH,100)');
            CA = sum(hist(cA,100)');
            CV = sum(hist(cV,100)');
            CD = sum(hist(cD,100)');
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
            [h,w] = size(img);
            [BGa,BGb] = detBG(img);
            [TGa,TGb] = detTG(img);
            [CGa,pbhist] = detCG(img);
            f = cell(1,40);
            cou = 1;
            for dr = 1:8,
                BGA = mean(sum(hist(BGa(:,:,dr),128)'));
                f{1,cou} = BGA;
                cou = cou + 1;
                
                
                TGA = mean(sum(hist(TGa(:,:,dr),128)'));
                f{1,cou} = TGA;
                cou = cou + 1;
                                
                for color = 1:3,
                    COLRS = mean(sum(hist(CGa(:,:,color,dr),128)'));
                    f{1,cou} = COLRS;
                    cou = cou + 1;
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

function [l, f] = rgb(nonzeroimpix)
            %RGBhistogram  2427-2626
            h= (hist(nonzeroimpix, 10));
            avg = (abs(h - mean(h)));
            [i,f] = min(avg);
            l = size(f);
end

function [l, f] = rgf(imggg)
            %region properties
            try
            regprops = regionprops(imggg,'all');
            elements = numel(regprops);
            %number of regions from BW level image
            f = double(elements);%/nonzerosum;
            catch
                f = 0;
            end
            l = size(f);
end

function [l, f] = rbf (imgbw)
            %region properties
            try
            regprops = regionprops(imgbw,'all');
            elements = numel(regprops);
            %number of regions from grey level image
            f = double(elements);%/nonzerosum;
            catch
                f = 0;
            end
            l= size(f);

end
