function[ac] = acutance(im)
%function[acutance] = acutance(im)
%
% Computes Acutance
% arguments:
%       im - image

%if (isrgb(im))
%    im = rgb2grey(im);
%end;
%if (~isfloat(im))
%    im = im2double(im);
%end;

[h,w,l] = size(im);

    %acutance
    gdiff = 0.0;
    %average image
    iav = zeros(h,w);
    for i = 2:h
        iav(h,1) = xor(iav(h-1,1),1);
    end
    for j = 2:w
        iav(:,j) = xor(iav(:,j-1),1);
    end
    iav = iav*255;
    iav_gdiff = 0.0;
    for i = 2:h-1
        for j = 2:w-1
            diff = 0;
            diff = diff + (iav(i,j) - iav(i,j-1))^2;
            diff = diff + (iav(i,j) - iav(i,j+1))^2;
            diff = diff + (iav(i,j) - iav(i-1,j-1))^2;
            diff = diff + (iav(i,j) - iav(i-1,j+1))^2;
            diff = diff + (iav(i,j) - iav(i+1,j-1))^2;
            diff = diff + (iav(i,j) - iav(i+1,j+1))^2;
            diff = diff + (iav(i,j) - iav(i-1,j))^2;
            diff = diff + (iav(i,j) - iav(i+1,j))^2;
            iav_gdiff = iav_gdiff + diff/8;
        end;
    end;
    iav_diff = iav_gdiff/((h-2)*(w-2));
    
    gdiff = 0.0;
    for i = 2:h-1
        for j = 2:w-1
            diff = 0;
            diff = diff + (im(i,j) - im(i,j-1))^2;
            diff = diff + (im(i,j) - im(i,j+1))^2;
            diff = diff + (im(i,j) - im(i-1,j-1))^2;
            diff = diff + (im(i,j) - im(i-1,j+1))^2;
            diff = diff + (im(i,j) - im(i+1,j-1))^2;
            diff = diff + (im(i,j) - im(i+1,j+1))^2;
            diff = diff + (im(i,j) - im(i-1,j))^2;
            diff = diff + (im(i,j) - im(i+1,j))^2;
            gdiff = gdiff + diff/8;
        end;
    end;
    gdiff = gdiff /((h-2)*(w-2));
    ac = 10^4*gdiff/(iav_diff);
    