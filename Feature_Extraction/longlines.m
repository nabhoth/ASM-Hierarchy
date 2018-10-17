function[lines, pixels] = longlines(ims, ime, thresh, radius, length)
%Detect lines of a given length by joinging pixels within a radius
%
%function[lines, pixels] = longlines(im_sal, im_edge, thresh, radius, length)
%
%parameters:
%       im_sal - image salience map
%       im_edge - image edge detectors
%       thresh - threshold for filtering edges through the salience 
%       radius - the radius for joing points within a line
%       length - the minimal length og lines to be kept

im_sal = ims;%imread(ims);
im_edge = ime;%imread(ime);
%if isrgb(im_sal), im_sal =rgb2gray(im_sal);end
%if isrgb(im_edge), im_edge =rgb2gray(im_edge);end


[im, points, ipoints] = thresh_edge(ims, ime,thresh);
l = size(points);
cols = l(2);

%imshow(im_sal);
%hold on;
iml = size(im_sal);
pixels = zeros(iml,'uint8');
dimension = uint8(l(2)/length);
lines = zeros(dimension,250,3);
llines = zeros(dimension,1);


size(pixels);
line_counter = 1;
counter = 1;
for c=1:1:cols
    if (pixels(points(2,c), points(1,c)) < 128), 
        if counter == 1,
            xs = zeros(1,length*100);
            ys = zeros(1,length*100);
            is = zeros(1,length*100);
            xs(1,counter) = points(1,c);
            ys(1,counter) = points(2,c);
            is(1,counter) = im_sal(points(2,c),points(1,c));
            counter = counter + 1;
            current_point = [points(1,c) points(2,c)];
        end
        for cc=1:1:cols
            if(pixels(points(2,cc), points(1,cc)) < 128), 
                    if (((points(1,cc) < (current_point(1) +radius)) && (points(1,cc) > (current_point(1) -radius))) && ((points(2,cc) < (current_point(2) +radius)) && (points(2,cc) > (current_point(2) -radius)))),
                        xs(1,counter) = points(1,cc);
                        ys(1,counter) = points(2,cc);
                        is(1,counter) = im_sal(points(2,cc),points(1,cc));
                        %pixels(points(2,cc), points(1,cc)) = 128;
                        counter = counter + 1;
                        current_point = [points(1,cc) points(2,cc)];
                    end
            end
        end
        if(counter > length),
            for l=1:counter-1,
                pixels(ys(1,l), xs(1,l)) = 128;
            end
            %pixels(ys(1,1:counter-1), xs(1,1:counter-1)) = 128;
            %llines(line_counter,:) = line(xs(1,1:counter-1),ys(1,1:counter-1));
           lines(line_counter,1,1) = -1;
           lines(line_counter,1,2) = counter-1;
           lines(line_counter,2:counter,1:3) = [xs(1,1:counter-1)',ys(1,1:counter-1)',is(1,1:counter-1)'];
           line_counter = line_counter + 1;
        end
        counter = 1;
    end
end
%size(pixels)

pixels1 = zeros(iml,'uint8');
%size(pixels1)
counter = 1;
for c=1:1:cols
    if (pixels1(ipoints(2,c), ipoints(1,c)) < 128), 
        if counter == 1,
            xs = zeros(1,length*100);
            ys = zeros(1,length*100);
            is = zeros(1,length*100);
            xs(1,counter) = ipoints(1,c);
            ys(1,counter) = ipoints(2,c);
            is(1,counter) = im_sal(points(2,c),points(1,c));
            counter = counter + 1;
            current_point = [ipoints(1,c) ipoints(2,c)];
        end
        for cc=1:1:cols
            if(pixels1(ipoints(2,cc), ipoints(1,cc)) < 128), 
                    if (((ipoints(1,cc) < (current_point(1) +radius)) && (ipoints(1,cc) > (current_point(1) -radius))) && ((ipoints(2,cc) < (current_point(2) +radius)) && (ipoints(2,cc) > (current_point(2) -radius)))),
                        xs(1,counter) = ipoints(1,cc);
                        ys(1,counter) = ipoints(2,cc);
                        is(1,counter) = im_sal(points(2,cc),points(1,cc));
                        %pixels1(ipoints(2,cc), ipoints(1,cc)) = 128;
                        counter = counter + 1;
                        current_point = [ipoints(1,cc) ipoints(2,cc)];
                    end
            end
        end
        if(counter > length),
            for l=1:counter-1,
                pixels1(ys(1,l), xs(1,l)) = 128;
            end
%           pixels1(ys(1,1:counter-1), xs(1,1:counter-1)) = 128;
           lines(line_counter,1,1) = -1;
           lines(line_counter,1,2) = counter-1;
           lines(line_counter,2:counter,1:3) = [xs(1,1:counter-1)',ys(1,1:counter-1)',is(1,1:counter-1)'];

           %llines(line_counter,:) = line(xs(1,1:counter-1),ys(1,1:counter-1));
           line_counter = line_counter + 1;
        end
        counter = 1;
    end
end
%size(pixels1)