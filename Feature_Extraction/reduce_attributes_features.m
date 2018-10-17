function [mapping] = reduce_attributes_features(bins,removal)
%DefaultVal('*removal', 'true');

%removes attributes regions that are not suitable for learning

inpath = './Image_Understanding/data/JPEGImages/%s';
%mappath = '/mnt/images/2/VOC2012/Regions/Textures/%s';
%out_path = '/mnt/images/2/VOC2012/Regions/%s';
%ins = 'Textures_Hypothesis_Regions_Mapping_BN.mat';
ins = 'Features_Hypothesis_Regions_Mapping_BN_Test.mat';
%load(sprintf(mappath, ins), 'mapping');
load(sprintf(inpath, ins), 'mapping');
%outs = ['Textures_Hypothesis_Regions_Mapping_BN_%d_reduced_' removal '.mat'];
if bins > 0
	outs = ['Features_Hypothesis_Regions_Mapping_BN_Test_%d_reduced_' removal '.mat'];
	outs = sprintf(outs, bins);
else
	outs = ['Features_Hypothesis_Regions_Mapping_BN_NOBINS_Test_reduced_' removal '.mat'];
	outs = sprintf(outs, bins);
end
names = mapping(1,6:end);

%[r,c] = size(mapping(1:end,:);
% attribNames={'Hypothesis','Area', 'Centroid1', 'Centroid2', 'BoundingBox', 'ConvexArea', 'Eccentricity', 'EulerNumber', 'Extent', 'MajorAxis', 'MinorAxis', 'Major/Minor', 'Orientation', 'Solidity'};
% attributes = cell(numel(classnames)+1, 14);
% attributes(1,:) = attribNames;
% attributes(2:end,1) = classnames;
[r,c] = size(mapping(2:end, 6:end));
tempmap = zeros(r,c);
binranges = zeros(numel(names),3);

for j=numel(names):-1:1
j
names{1,j}
    t = cell2mat(mapping(2:end,j+5));
    if bins > 0	
	    mx = max(t(:,1));
	    mi = min(t(:,1));
	    if mx > mi
        	binranges(j,1:bins) = linspace(mi,mx,bins);
	        [~,temp] = histc(t,binranges(j,:));
        	tempmap(:,j) = temp;
	    else
        	binranges(j,:) = [];
	        tempmap(:,j) = [];
        	mapping(:,j+5) = [];
	    end
    else
	tempmap(:,j) = t;
    end
end
% for i=1:numel(classnames)
%    filename = sprintf(inpath, classnames{i});
%    load(filename, 'map');
%    map = map(map(:,4) > 0.001,:);
%    replac = sprintf(inpath, strcat(classnames{i},'_Minimized'));
%    save(replac, 'map', '-v7.3')
%    
%    map(:,15) = (map(:,15)+90)./180;
%    
%    max10 = max(map(:,10));
%    
%    map(:,10) = map(:,10)./max10;
%    
%    attr = mean(map,1);
%    attr = attr(:,4:end);
%    
%    replac = sprintf(inpath, strcat(classnames{i},'_BN'));
%    save(replac, 'attr', '-v7.3')
%    
%    tempmap(i,:) = attr;
%    
% end
% for i=1:13
%     mx = max(tempmap(:,i));
%     mi = min(tempmap(:,i));
%    [~,tempmap(:,i)] = histc(tempmap(:,i),linspace(mi,mx,10));
% end
       
[r,c] = size(mapping(2:end, 6:end))
outpath = sprintf(inpath, outs)
C = num2cell(tempmap);%, ones(r,1), ones(c,1));
mapping(2:end,6:end) =  C;



%this is a hack
if strcmp(removal, 'true')
	mapping (:,22:24) = [];
	mapping (:,9:17) = [];
end
save(outpath, 'mapping', '-v7.3');

if bins > 0
	%save the binranges
	[br,bc] = size(binranges)
	C = cell(br, bc+1);
	C(:,2:end) = num2cell(binranges);
	if strcmp(removal, 'true')
		C(22:24,:) = [];
		C(9:17,:) = [];
	end
	C(:,1) = mapping(1,6:end)';
	binranges = C;

	outpath = sprintf(inpath, ['reduced_' removal, 'binranges_' sprintf('%d', bins) '.mat'])
	save(outpath, 'binranges', '-v7.3');
end

end

