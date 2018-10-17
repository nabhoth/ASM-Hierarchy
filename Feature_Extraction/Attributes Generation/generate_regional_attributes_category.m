function [] = generate_regional_attributes_category(  )
% prepare training/testing data for training/testing of algorithms 
% for each region corresponding to a particular hypothesis represented by a
% bounding box (approximate hypothesis) features and attributes are extracted 



clear all;
VOCinit;

% image test set
sprintf(VOCopts.seg.imgsetpath,VOCopts.trainset)

[gtids,t]=textread(sprintf(VOCopts.seg.imgsetpath,VOCopts.trainset),'%s %d');
%output path
output_path = '/mnt/images/ASM-data/data/Categories/';
%mkdir('/mnt/images/2/VOC2012/', 'Categories/');
%nclass = VOCopts.classes+1;
%confcounts = zeros(num);

%train_data = zeros(length(gtids),(11+69));

[mapping{1:VOCopts.nclasses}] = deal([]);

for n=1:length(gtids)
    
    counter = 1;
    imname = gtids{n};
try
    %read input image
    infile = sprintf(VOCopts.imgpath,imname)
    [inim] = imread(infile);
    [h,w] = size(inim);
    
    %read ground truth
    gtfile = sprintf(VOCopts.seg.clsimgpath,imname);
    [gtim] = imread(gtfile);
    gtim = double(gtim);
    
    %read salience scaled
    sfile = sprintf(VOCopts.imgpath,['salience/',imname]);
    sfile =  [sfile(1:end-4) '_salience.jpg'];
    [sim] = imread(sfile);
    [sh,sw] = size(sim);
    if sh ~= h
        sim = imresize(sim, [h w]) ;
    end
    sim = double(sim);
    
    %read edge image
    efile = sprintf(VOCopts.imgpath,['edge/', imname]);
    efile = [efile(1:end-4) '_edge.ppm']
    [eim] = imread(efile);
    [eh,ew] = size(eim);
    if eh ~= h
        eim = imresize(eim, [h w]) ;
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
        year = str2double(jgtids(1:4))
        ids = str2double(jgtids(6:end))

        [x] = find(strcmp(recs.objects(j).class, VOCopts.classes));
%        out_names{all_count} = jgtids;
        % extract objects of class
        gt=struct('BB',[],'diff',[],'det',[]);
        gt.BB=cat(1,recs.objects(j).bbox)';
        gt.diff=[recs.objects(j).difficult];
        gt.det=false(length(j),1);
        inimsub = inim(recs.objects(j).bbox(2):recs.objects(j).bbox(4), recs.objects(j).bbox(1):recs.objects(j).bbox(3),:);
        eimsub = eim(recs.objects(j).bbox(2):recs.objects(j).bbox(4), recs.objects(j).bbox(1):recs.objects(j).bbox(3),:);
        simsub = sim(recs.objects(j).bbox(2):recs.objects(j).bbox(4), recs.objects(j).bbox(1):recs.objects(j).bbox(3),:);

        %try
            mapping{x} = [mapping{x};find_best_features_normalized_single(year, ids, j, inimsub, simsub, eimsub)];
        %catch
        %    sprintf('error')
        %end        
    end
   catch
	sprintf('error')
   end

 
end
for i=1:VOCopts.nclasses
    map = mapping{i};
    %save(strcat(output_path,'/',sprintf('%s', VOCopts.classes{i}), '_Features_Hypothesis_Regions_Mapping.mat'), 'map', '-v7.3');
    save(strcat(output_path,'/',sprintf('%s', VOCopts.classes{i}), '_Attributes_Hypothesis_Regions_Mapping.mat'), 'map', '-v7.3');
end



