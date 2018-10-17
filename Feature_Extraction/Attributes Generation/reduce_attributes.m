function [] = reduce_attributes()
%removes attributes regions that are not suitable for learning
path = '/mnt/images/ASM-data/data/Categories/%s_Attributes_Hypothesis_Regions_Mapping.mat';


classnames = {'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', ...
    'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', ...
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'};

attribNames={'Hypothesis','Area', 'Centroid1', 'Centroid2', 'BoundingBox', 'ConvexArea', 'Eccentricity', 'EulerNumber', 'Extent', 'MajorAxis', 'MinorAxis', 'Major/Minor', 'Orientation', 'Solidity'};
attributes = cell(numel(classnames)+1, 14);
attributes(1,:) = attribNames;
attributes(2:end,1) = classnames;
tempmap = zeros(numel(classnames), 13);

binranges = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1];

for i=1:numel(classnames)
   filename = sprintf(path, classnames{i});
   load(filename, 'map');
   map = map(map(:,4) > 0.001,:);
   replac = sprintf(path, strcat(classnames{i},'_Minimized'));
   save(replac, 'map', '-v7.3')
   
   map(:,15) = (map(:,15)+90)./180;
   
   max10 = max(map(:,10));
   
   map(:,10) = map(:,10)./max10;
   
   attr = mean(map,1);
   attr = attr(:,4:end);
   
   replac = sprintf(path, strcat(classnames{i},'_BN'));
   save(replac, 'attr', '-v7.3')
   
   tempmap(i,:) = attr;
   
end
for i=1:13
    mx = max(tempmap(:,i));
    mi = min(tempmap(:,i));
   [~,tempmap(:,i)] = histc(tempmap(:,i),linspace(mi,mx,10));
end
       

replac = sprintf(path, 'Grouped');
C = mat2cell(tempmap, ones(numel(classnames),1), ones(13,1));
attributes(2:end,2:end) =C;
save(replac, 'attributes', '-v7.3')
end

