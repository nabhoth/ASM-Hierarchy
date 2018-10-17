function [mapping] = get_features_bn(im, flabels)
%function [ mapping] = get_features_bn(im)

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
            k = numel(flabels);
            mapping = cell(1,k); 
            %mapping = cell(1,7);
%            nonzeroimpix = I(I.* repmat(mask,[1,1,3]) > 0);
            
            IG = rgb2gray(I);
            graynonzeroim = IG(IG > 0);

            IBW = im2bw(I);
            %imrgb = im2double(im);
            %imBW = im2double(IBW);
            imgg = im2double(IG);
            [h,w] = size(IG);
            
          
            fprintf('processing...\n');
            count = 1;
%need to add cases
  for i = 1:k
       
      if (strcmp(flabels(i),'brightness'))
          [lh, f] = his(graynonzeroim)
            mapping{1,i} = f(1,1);
      end
      
      if (strcmp(flabels(i),'fft'))            
            [lg, f] = ft(imgg);
            mapping{1, i} = f;            
      end
      
       if (strcmp(flabels(i),'Gabor edges'))  
            [lg, f] = gab(IG);
            mapping{1, i} = f;            
       end
       
       if (strcmp(flabels(i),'wavelets1'))     
            [lg, f] = wav(IG);
            mapping{1, i} = f{1,1};           
       end
       
       if (strcmp(flabels(i),'wavelets2'))     
            [lg, f] = wav(IG);
            mapping{1, i} = f{1,2};            
       end
       
       if (strcmp(flabels(i),'wavelets3'))     
            [lg, f] = wav(IG);
            mapping{1, i} = f{1,3};            
       end       
       
       if (strcmp(flabels(i),'wavelets4'))  
            [lg, f] = wav(IG);
            mapping{1, i} = f{1,4};            
       end
       
        if (strcmp(flabels(i),'contrast'))   
            [lg, f] = con(imgg);
            mapping{1, count} = f;
            count = count + lg;
        end

        if (strcmp(flabels(i),'accutance')) 
            [lg, f] =acu(IG);
            mapping{1, i} = f;
        end
        
        if (strcmp(flabels(i),'gist')) 
            [lg, f] = gis(IG);
            mapping{1, i} = f;            
        end
        
        if (strcmp(flabels(i),'colorR')) 
            [lb, f] = rgb(im);
            mapping{1, i} = f{1};
        end 
         
         if (strcmp(flabels(i),'colorG')) 
            [lb, f] = rgb(im);
            mapping{1, i} = f{2};
         end 
          
         if (strcmp(flabels(i),'colorB')) 
            [lb, f] = rgb(im);
            mapping{1, i} = f{1};
         end 
         
         if (strcmp(flabels(i),'grey regions1')) 
            [lr, f] = rgf(IG);
            mapping{1, i} = f{1};
         end
         
         if (strcmp(flabels(i),'grey regions2')) 
            [lr, f] = rgf(IG);
            mapping{1, i} = f{2};
         end
         
         if (strcmp(flabels(i),'grey regions3')) 
            [lr, f] = rgf(IG);
            mapping{1, i} = f{3};
         end
         
         if (strcmp(flabels(i),'grey regions4')) 
            [lr, f] = rgf(IG);
            mapping{1, i} = f{4};
         end
            
         if (strcmp(flabels(i),'bw regions1')) 
            [lrr, f] = rbf (IBW);
             mapping{1, i} = f{1};
         end
         
         if (strcmp(flabels(i),'bw regions2')) 
            [lrr, f] = rbf (IBW);
             mapping{1, i} = f{2};
         end
         
         if (strcmp(flabels(i),'bw regions3')) 
            [lrr, f] = rbf (IBW);
            mapping{1, i} = f{3};
         end
         
         if (strcmp(flabels(i),'bw regions4')) 
            [lrr, f] = rbf (IBW);
            mapping{1, i} = f{4};
         end
         
         
   end
          
            
            %{
            [lh, f] = his(graynonzeroim)
            mapping{1, count} = f(1,1)
            count = count + lh
            lh
            
            [lg, f] = ft(imgg);
            mapping{1, count} = f;
            count = count + lg;

            [lg, f] = gab(IG);
            mapping{1, count} = f;
            count = count + lg;

            [lg, f] = wav(IG);
            mapping(1, count:count+lg-1) = f;
            count = count + lg;

            [lg, f] = con(imgg);
            mapping{1, count} = f;
            count = count + lg;

            [lg, f] =acu(IG);
            mapping{1, count} = f;
            count = count + lg;

            [lg, f] = gis(IG);
            mapping{1, count} = f;
            count = count + lg;

            [lb, f] = rgb(im);
            mapping(1, count:count+lb-1) = f;
            count = count + lb;

            [lr, f] = rgf(IG);
            mapping(1, count:count+lr-1) = f;
            count = count + lr;
            
            [lrr, f] = rbf (IBW);
            mapping(1, count:count+lrr-1) = f;
            count = count + lrr;
           %}
          

end

function [l, f] = his(imgg)
            %sum of gray histogram
            H = imhist(imgg, 10); %min-max
            avg = (abs(H - mean(H)));
            [i,f] = (min(avg));
            l = 1;
end            
            
function [l, f] = ft(imgg)           
            %sum of fft
            try
            histo = imhist(fft(imgg,10),10);
            avg = (abs(histo - mean(histo)));
            [i,f] = (min(avg));
            catch
                f = 0;
            end
            l = 1;
end

function [l, f] = gab(imgg)         
            %gabor filter
            [G,gabout] = gaborfilter1(imgg(:,:,1),2,4,16,4*pi/3);
            histo = hist(reshape(gabout, prod(size(gabout)), 1),10);
            avg = abs(histo - mean(histo));
            [i,f] = (min(avg));
            l = 1;
end

function [l, f]=wav(imgg)          
            %wavelet
            [cA,cH,cV,cD] = (dwt2(imgg,'haar'));
            CH = imhist(cH,100);
            CH = (abs(CH - mean(CH)));
            CA = imhist(cA,100);
            CA = (abs(CA - mean(CA)));
            CV = imhist(cV,100);
            CV = (abs(CV - mean(CV)));
            CD = imhist(cD,100);
            CD = (abs(CD - mean(CD)));
            f = cell(1,4);
            f{1,1} = mean(CH);
            f{1,2} = mean(CA);
            f{1,3} = mean(CV);
            f{1,4} = mean(CD);
            l = 4;
end

function [l, f] = con(imgg)
            %contrast
            f = contrast(imgg)/10^4;
            l = 1;
end

function [l, f] = acu(imgg)
            %acutance
            f = acutance(imgg)*10^4;
            l = 1;
end
function [l, f] = gis(img) 
            %Gist
            h = gist_features(img);
            h = abs(h - mean(h));
            f = mean(h);
            l = 1;
end




function [l, f] = rgb(nonzeroimpix)
            %RGBhistogram  2427-2626
            r= imhist(nonzeroimpix(:,:,1), 10);
            avgr = abs(r - mean(r));
            g= imhist(nonzeroimpix(:,:,2), 10);
            avgg = abs(g - mean(g));
            b= imhist(nonzeroimpix(:,:,3), 10);
            avgb = abs(b - mean(b));
            f = {min(avgr), min(avgg), min(avgb)};
            l = 3;
end

function [l, f] = rgf(imggg)
            %grey region properties
%            try
            regprops = regionprops(imggg,'Area');
            %h = hist([regprops.Area]);
            [r,c] = size(regprops);
            if r > 0
                h = [regprops.Area];
                %avg = abs(h - mean(h));
                %elements = numel(regprops);
                %number of regions from BW level image
                %f = double(elements);%/nonzerosum;
                f = {min(h), max(h), mean(h), std(h)};
            else
                f = {0,0,0,0};
            end
            l = 4;
end

function [l, f] = rbf (imgbw)
            % bw region properties
 %           try
            regprops = regionprops(imgbw,'Area');
            %h = hist([regprops.Area]);
            [r,c] = size(regprops);
            if r > 0
                h = [regprops.Area];
                %avg = abs(h - mean(h));
                %elements = numel(regprops.Centroid)
                %number of regions from grey level image
                %f = double(elements);%/nonzerosum;
                f ={min(h), max(h), min(h), max(h)};
            else
                f = {0,0,0,0}
            end
                l= 4;

end


function [text_k] = texture(inimsub)
    [~,~,tmap] = detTG(inimsub);
    text_k =zeros(1,64);
    for i=1:64
        text_k(1,i) = numel(tmap(tmap == i));
    end
end

