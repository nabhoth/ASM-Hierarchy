function [] = generate_regional_data_for_bn_bb(  )
% prepare training/testing data for training/testing of algorithms 
% for each region corresponding to a particular hypothesis represented by a
% bounding box (approximate hypothesis) features and attributes are extracted 



clear all;
VOCinit;

% image test set
[gtids,t]=textread(sprintf(VOCopts.seg.imgsetpath,VOCopts.testset),'%s %d');
%output path
output_path = '/mnt/images/2/VOC2012/Regions/';

%nclass = VOCopts.classes+1;
%confcounts = zeros(num);

%train_data = zeros(length(gtids),(11+69));

mapping = [];
for n=1:length(gtids)
    try 
    counter = 1;
    imname = gtids{n};

    %read input image
    infile = sprintf(VOCopts.imgpath,imname);
    [inim] = imread(infile);
    [h,w,l] = size(inim)
    
    %read ground truth
    gtfile = sprintf(VOCopts.seg.clsimgpath,imname);
    [gtim] = imread(gtfile);
    gtim = double(gtim);
    
    %read salience scaled
    sfile = sprintf(VOCopts.imgpath,imname);
    sfile =  [sfile(1:end-4) '_salience.jpg']
    [sim] = imread(sfile);
    [sh,sw,l] = size(sim)
    if sh ~= h
        sim = imresize(sim, [h w]);
    end
    %sim = double(sim);
    
    %read edge image
    efile = sprintf(VOCopts.imgpath,imname);
    efile = [efile(1:end-4) '_edge.ppm'];
    [eim] = imread(efile);
    [eh,ew,l] = size(eim);
    if eh ~= h
        eim = imresize(eim, [h w]);
    end    
    eim = double(eim);
    
    
    %get the bounding box for all objects
    % read annotation
    recs=PASreadrecord(sprintf(VOCopts.annopath,gtids{n}));

    
    for j=1:numel(recs.objects)
        if (j > 9)
            jgtids = strrep(gtids{n}, '_00', sprintf('_%d',j));
        else
            jgtids = strrep(gtids{n}, '_00', sprintf('_0%d',j));            
        end
        year = str2double(jgtids(1:4));
        ids = str2double(jgtids(6:end));
        
%        out_names{all_count} = jgtids;
        % extract objects of class
        gt=struct('BB',[],'diff',[],'det',[]);
        gt.BB=cat(1,recs.objects(j).bbox)';
        gt.diff=[recs.objects(j).difficult];
        gt.det=false(length(j),1);
%        outfile = [output_path gtids{n} '_' sprintf('%d',counter) '-' recs.objects(j).class '.jpg'];
%        gtimsub = gtim(recs.objects(j).bbox(2):recs.objects(j).bbox(4), recs.objects(j).bbox(1):recs.objects(j).bbox(3),:);
        inimsub = inim(recs.objects(j).bbox(2):recs.objects(j).bbox(4), recs.objects(j).bbox(1):recs.objects(j).bbox(3),:);
        eimsub = eim(recs.objects(j).bbox(2):recs.objects(j).bbox(4), recs.objects(j).bbox(1):recs.objects(j).bbox(3),:);
        simsub = sim(recs.objects(j).bbox(2):recs.objects(j).bbox(4), recs.objects(j).bbox(1):recs.objects(j).bbox(3),:);

        try
            mapping = [mapping;find_best_features_normalized_single(year, ids, j, inimsub, simsub, eimsub)];
        catch
            
        end        
%         [ho,wo] = size(gtimsub);
%         hscale = 300/ho;
%         wscale = 300/wo;
%         if (ho <  250)
%             outim = imresize(gtimsub, hscale);
%         else if (wo < 250)
%                 outim = imresize(gtimsub, wscale);
%             end
%         end
%         save(strcat(dir_path,'/', prefx,'_Features_Hypothesis_Regions_Mapping.mat'), 'feats', '-v7.3');
%            
% %        imshow(outim)
jgtids
        outfile = [output_path jgtids '.jpg'];
        imwrite(inimsub, outfile,'JPG');
        outfile = [output_path jgtids '_edge.jpg'];
        imwrite(eimsub, outfile,'JPG');
        outfile = [output_path jgtids '_salience.jpg'];
        imwrite(simsub, outfile,'JPG');
%         counter = counter + 1;
%         all_count = all_count + 1;
    end
    
    catch
        
    end
    
end
save(strcat(output_path,'/Features_Hypothesis_Regions_Mapping_Test.mat'), 'mapping', '-v7.3');



