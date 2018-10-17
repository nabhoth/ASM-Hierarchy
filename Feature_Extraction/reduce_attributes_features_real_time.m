function [reduced] = reduce_attributes_features_real_time(binranges, data)
% reduces the real time data (vector) of either attributes or features to
% histogram with binranges obtained from the learning data

[~,c] = size(data);
[~,bc] = size(binranges);
reduced = cell(1,c);
for j=1:c-3
    if ~isempty(data{j})
        for a=2:bc
            if a < bc
                if data{j} >= binranges{j,a-1} & data{j} < binranges{j,a}
                    reduced{j} = a-1;
%                    continue;
                else
                    if data{j} >= binranges{j,a}
                        reduced{j} = a;
                    end
                end
            end
        end
    end
end


