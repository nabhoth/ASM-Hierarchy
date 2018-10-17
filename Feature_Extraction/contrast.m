function [ ct ] = contrast( im )
%calculates contrast of an image
%if (isrgb(im))
%    im = rgb2grey(im);
%end;
%if (~isfloat(im))
%    im = im2double(im);
%end;
[h,w,l] = size(im);
%histogram of the gray scale
    histo = hist(im, 256);
    %do cumulative histogram
    %first sum values for all columns
    for i = 2:w,
        histo(:,i) = histo(:,i)+histo(:,i-1);
    end
    h1 = histo(:,w);
    %cumulatively sum all rows(bins)
    for i = 2:256,
        h1(i,:) = h1(i,:)+h1(i-1,:);
    end
    h2 = h1;
    %h = 
    X = zeros(2,256);
    for i = 1:256,
        X(1,i) = 1; X(2,i) = i;
    end;
    tX = X';
    a1 = 0;a2 = 0;a3 = 0;a4 = 0;
    for i = 1:256,
        a1 = a1 + (tX(i,1)*X(1,i));a2 = a2 + (tX(i,1)*X(2,i));a3 = a3 + (tX(i,2)*X(1,i));a4 = a4 + (tX(i,2)*X(2,i));
    end;

    b1 = 0;b2 = 0;
    for i = 1:256,
        b1 = b1 + tX(i,1)*h2(i);
        b2 = b2 + tX(i,2)*h2(i);
    end;

    c1 = (a1*b2-a3*b1)/(a1*a4-a2*a3);
    c0 = (b1-a2*(c1))/a1;

    rms = 0.0;
    for i = 1:256,
        rms = rms + (h2(i)-(c0-c1*i))^2;
    end;

    rms = rms/256;
    ct = sqrt(rms);
    
end

